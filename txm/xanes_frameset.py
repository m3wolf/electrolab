import os
from collections import defaultdict
import re
import math
import multiprocessing
import queue
import warnings

import pandas as pd
from matplotlib import pyplot
from matplotlib.colors import Normalize
from mapping.colormaps import cmaps
import h5py
import numpy as np
from skimage import morphology, filters, feature, transform

from utilities import display_progress, xycoord
from .frame import TXMFrame, average_frames, calculate_particle_labels
from .gtk_viewer import GtkTxmViewer
from plots import new_axes
import exceptions
from hdf import HDFAttribute
import smp


def build_dataframe(frames):
    index = [frame.energy for frame in frames]
    images = [pd.DataFrame(frame.image_data) for frame in frames]
    series = pd.Series(images, index=index)
    return series

class XanesFrameset():
    _attrs = {}
    active_groupname = None # HDFAttribute('active_groupname')
    latest_groupname = HDFAttribute('latest_groupname')
    background_groupname = HDFAttribute('background_groupname')
    active_particle_idx = HDFAttribute('active_particle_idx', default=None,
                                       group_func='active_group')
    latest_labels = HDFAttribute('latest_labels', default='particle_labels')
    map_name = HDFAttribute('map_name')
    cmap = 'plasma'
    def __init__(self, filename, groupname, edge=None):
        self.hdf_filename = filename
        self.parent_groupname = groupname
        self.edge = edge
        # Check to make sure a valid group is given
        with self.hdf_file() as hdf_file:
            if not groupname in hdf_file.keys():
                msg = "Created new frameset group: {}"
                print(msg.format(groupname))
                hdf_file.create_group(groupname)
        self.active_groupname = self.latest_groupname

    def __iter__(self):
        """Get each frame from the HDF5 file"""
        hdf_file = self.hdf_file()
        for dataset_name in self.active_group().keys():
            yield TXMFrame.load_from_dataset(self.active_group()[dataset_name])

    def __len__(self):
        return len(self.active_group().keys())

    def __getitem__(self, index):
        hdf_file = self.hdf_file()
        # First just use the index directly
        try:
            frame = TXMFrame.load_from_dataset(self.active_group()[index])
        except AttributeError:
            name = list(self.active_group().keys())[index]
            frame = TXMFrame.load_from_dataset(self.active_group()[name])
        return frame

    @property
    def active_labels_groupname(self):
        """The group name for the latest frameset of detected particle labels."""
        # Save as an HDF5 attribute
        group = self.active_group()
        return group.attrs.get('active_labels', None)

    @active_labels_groupname.setter
    def active_labels_groupname(self, value):
        group = self.active_group()
        group.attrs['active_labels'] = value

    @active_labels_groupname.deleter
    def active_labels_groupname(self):
        group = self.active_group()
        del group.attrs['active_labels']

    def particle(self, particle_idx=0):
        """Prepare a particle frameset for the given particle index."""
        fs = ParticleFrameset(parent=self, particle_idx=particle_idx)
        return fs

    def switch_group(self, name):
        if not name in self.hdf_group().keys():
            msg = "{name} is not a valid group. Choices are {choices}."
            raise exceptions.GroupKeyError(
                msg.format(name=name, choices=list(self.hdf_group().keys()))
            )
        self.active_groupname = name

    def fork_group(self, name):
        """Create a new, copy of the current active group inside the HDF
        parent with name: `name`.
        """
        # Create an empty group
        try:
            del self.hdf_group()[name]
        except KeyError:
            pass
        # Copy the old data
        # dest = self.hdf_group().create_group(name)
        # print(self.active_groupname, dest)
        self.hdf_group().copy(source=self.active_groupname, dest=name)
        # self.hdf_group().copy(source=self.active_groupname, dest=name)
        # Update the group name
        self.latest_groupname = name
        self.switch_group(name)

    def fork_labels(self, name):
        # Create a new group
        if name in self.hdf_group().keys():
            del self.hdf_group()[name]
        self.hdf_group().copy(self.active_labels_groupname, name)
        labels_group = self.hdf_group()[name]
        # Update label paths for frame datasets
        for frame in self:
            key = frame.image_data.name.split('/')[-1]
            new_label_name = labels_group[key].name
            frame.particle_labels_path = new_label_name
        self.latest_labels = name
        self.active_labels_groupname = name
        return labels_group

    def subtract_background(self, bg_groupname):
        self.background_groupname = bg_groupname
        self.fork_group('absorbance_frames')
        bg_group = self.hdf_file()[bg_groupname]
        for energy in display_progress(self.active_group().keys(),
                                       "Applying reference corrections"):
            sample_dataset = self.active_group()[energy]
            bg_dataset = bg_group[energy]
            new_data = np.log10(bg_dataset.value/sample_dataset.value)
            sample_dataset.write_direct(new_data)

    def correct_magnification(self):
        """Correct for changes in magnification at different energies.

        As the X-ray energy increases, the focal length of the zone
        plate changes and so the image is zoomed-out at higher
        energies. This method applies a correction to each frame to
        make the magnification similar to that of the first frame.
        """
        # Fork groups
        self.fork_group('magnified_frames')
        # Regression parameters
        slope=-0.00010558834052580277; intercept=1.871559636328671
        # Prepare multiprocessing objects
        def worker(payload):
            key, energy, data = payload
            # Determine degree of magnification required
            magnification = energy * slope + intercept
            original_shape = xycoord(x=data.shape[1], y=data.shape[0])
            # Expand the image by magnification degree and re-center
            translation = xycoord(
                x=original_shape.x/2*(1-magnification),
                y=original_shape.y/2*(1-magnification),
            )
            transformation = transform.SimilarityTransform(
                scale=magnification,
                translation=translation
            )
            # Apply the transformation
            result = transform.warp(data, transformation, order=3)
            return (key, energy, result)

        def process_result(payload):
            # Write the computed result back to disk
            key, energy, data = payload
            frame = self[key]
            frame.image_data.write_direct(data)

        queue = smp.Queue(worker=worker,
                          totalsize=len(self),
                          result_callback=process_result,
                          description="Correcting magnification")
        for frame in self:
            data = frame.image_data.value
            key = frame.image_data.name.split('/')[-1]
            energy = frame.energy
            # Add this frame to the queue for processing
            queue.put((key, energy, data))
        queue.join()

    def align_frames(self, particle_idx=None, particle_loc=None,
                     reference_frame=0):
        """Use phase correlation algorithm to line up the frames."""
        # Create new data groups to hold shifted image data
        self.fork_group('aligned_particle')
        self.fork_labels('aligned_labels')
        reference_image = self[reference_frame].image_data.value
        # Determine which particle to use
        if particle_idx is not None:
            # Apply a mask if a particle is specified
            self.active_particle_idx = particle_idx
            reference_mask = self[reference_frame].particles()[particle_idx].full_mask()
            reference_image = np.ma.array(reference_image, mask=reference_mask)
        elif particle_loc is not None:
            particle = self[reference_frame].activate_closest_particle(loc=particle_loc)
            reference_mask = particle.full_mask()
        # Multiprocessing setup
        def worker(payload):
            key, data, labels, mask = payload
            masked_data = np.ma.array(data, mask=mask)
            results = feature.register_translation(reference_image, masked_data)
            shift, error, diffphase = results
            shift = xycoord(-shift[1], -shift[0])
            # Apply net transformation
            transformation = transform.SimilarityTransform(translation=shift)
            new_data = transform.warp(data, transformation, order=3, mode="wrap")
            # Transform labels
            original_dtype = labels.dtype
            labels = labels.astype(np.float64)
            new_labels = transform.warp(labels, transformation, order=0, mode="wrap")
            new_labels = new_labels.astype(original_dtype)
            return (key, new_data, new_labels)
        def process_result(payload):
            key, data, labels = payload
            frame = self[key]
            frame.image_data.write_direct(data)
            frame.particle_labels().write_direct(labels)
            frame.activate_closest_particle(loc=particle_loc)
        queue = smp.Queue(worker=worker, result_callback=process_result,
                          totalsize=len(self), description="Aligning frames")
        for frame in self:
            key = frame.key()
            # Prepare data arrays
            data = frame.image_data.value
            labels = frame.particle_labels().value
            if particle_idx is not None:
                # Apply a mask if particle is specified
                particle = frame.particles()[particle_idx]
                mask = particle.full_mask()
            elif particle_loc is not None:
                particle = frame.activate_closest_particle(loc=particle_loc)
                mask = particle.full_mask()
            else:
                mask = np.zeros_like(data)
            # Launch transformation for this frame
            queue.put((key, data, labels, mask))
        queue.join()

    def align_to_particle(self, particle_loc=0):
        """Use the centroid position of given particle to align all the
        frames."""
        self.fork_group('aligned_frames')
        self.fork_labels('aligned_labels')
        # Determine average positions
        total_x = 0; total_y = 0; n=0
        for frame in display_progress(self, 'Computing true center'):
            particle = frame.activate_closest_particle(particle_loc)
            n+=1
            total_x += particle.centroid().x
            total_y += particle.centroid().y
        global_center = xycoord(x=total_x/n, y=total_y/n)
        # Align all frames to average position
        for frame in display_progress(self, 'Aligning frames'):
            particle = frame.particles()[frame.active_particle_idx]
            offset_x = int(round(global_center.x - particle.centroid().x))
            offset_y = int(round(global_center.y - particle.centroid().y))
            # Apply net transformation
            transformation = transform.SimilarityTransform(
                translation=(-offset_x, -offset_y))
            new_data = transform.warp(frame.image_data, transformation,
                                      order=3, mode="wrap")
            frame.image_data.write_direct(new_data)
            # Apply translation to particle labels
            labels = frame.particle_labels()
            new_labels = transform.warp(labels.value.astype(np.float64),
                                        transformation, order=3, mode="wrap")
            labels.write_direct(new_labels)
            # frame.shift_data(x_offset=offset_x,
            #                  y_offset=offset_y)
            # frame.shift_data(x_offset=offset_x,
            #                  y_offset=offset_y,
            #                  dataset=frame.particle_labels())

    def crop_to_particle(self):
        """Reduce the image size to just show the particle in question."""
        # Create new HDF5 groups
        self.fork_group('cropped_particle')
        self.fork_labels('cropped_labels')
        # Determine largest bounding box based on all energies
        boxes = [frame.particles()[frame.active_particle_idx].bbox()
                 for frame in self]
        left = min([box.left for box in boxes])
        top = min([box.top for box in boxes])
        bottom = max([box.bottom for box in boxes])
        right = max([box.right for box in boxes])
        # Roll each image to have the particle top left
        for frame in display_progress(self, 'Cropping frames'):
            frame.crop(top=top, left=left, bottom=bottom, right=right)
            # Determine new main particle index
            new_idx = np.argmax([p.convex_area() for p in frame.particles()])
            frame.active_particle_idx = new_idx

    def align_frame_positions(self):
        """Correct for inaccurate motion in the sample motors."""
        self.fork_group('aligned_frames')
        self.fork_labels('aligned_labels')
        # Determine average positions
        total_x = 0; total_y = 0; n=0
        for frame in display_progress(self, 'Computing true center'):
            n+=1
            total_x += frame.sample_position.x
            total_y += frame.sample_position.y
        global_x = total_x / n
        global_y = total_y / n
        for frame in display_progress(self, 'Aligning frames'):
            um_per_pixel_x = 40/frame.image_data.shape[1]
            um_per_pixel_y = 40/frame.image_data.shape[0]
            offset_x = int(round((global_x - frame.sample_position.x)/um_per_pixel_x))
            offset_y = int(round((global_y - frame.sample_position.y)/um_per_pixel_y))
            frame.shift_data(x_offset=offset_x, y_offset=offset_y)
            # Store updated position info
            new_position = (
                frame.sample_position.x + offset_x * um_per_pixel_x,
                frame.sample_position.y + offset_y * um_per_pixel_y,
                frame.sample_position.z
            )
            frame.sample_position = new_position

    def label_particles(self):
        labels_groupname = 'particle_labels'
        if labels_groupname in self.hdf_group().keys():
            del self.hdf_group()[labels_groupname]
        self.active_labels_groupname = labels_groupname
        # Create a new group
        labels_group = self.hdf_group().create_group(labels_groupname)
        # Callables for determining particle labels
        def worker(payload):
            key, data = payload
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                new_data = calculate_particle_labels(data)
            return (key, new_data)
        def process_result(payload):
            # Save the calculated data
            key, data = payload
            labels = self.hdf_group()[labels_groupname]
            dataset = labels.create_dataset(key, data=data, compression='gzip')
        # Prepare multirprocessing objects
        frame_queue = smp.Queue(worker=worker,
                                totalsize=len(self),
                                result_callback=process_result,
                                description="Detecting particles")
        for frame in self:
            data = frame.image_data.value
            key = frame.image_data.name.split('/')[-1]
            # process_result(worker((key, data)))
            frame_queue.put((key, data))
            # Write path to saved particle labels
            frame.particle_labels_path = labels_group.name + "/" + key
        # Wait for all processing to finish
        frame_queue.join()

    def rebin(self, shape=None, factor=None):
        """Resample all images into new shape. Arguments `shape` and `factor`
        passed to txm.frame.TXMFrame.rebin().
        """
        self.fork_group('rebinned')
        for frame in display_progress(self, "Rebinning"):
            frame.rebin(shape=shape, factor=factor)

    def plot_full_image(self):
        return pyplot.imshow(self.df.mean())

    def xanes_spectrum(self):
        """Collapse the dataset down to a two-d spectrum."""
        energies = []
        intensities = []
        for frame in self:
            data = frame.image_data.value
            # Find the active particle (if any)
            if frame.active_particle_idx is not None:
                # Frame-specific particle
                particle = frame.particles()[frame.active_particle_idx]
            elif self.active_particle_idx is not None:
                # Frameset global particle
                particle = frame.particles()[self.active_particle_idx]
            else:
                # No particle
                particle = None
            # Create mask that's the same size as the image
            if particle:
                bbox = particle.bbox()
                mask = np.zeros_like(data)
                mask[bbox.top:bbox.bottom, bbox.left:bbox.right] = particle.mask()
                mask = np.logical_not(mask)
                data[mask] = 0
            # Sum absorbances for datasets
            intensity = np.sum(data)/np.prod(data.shape)
            # Add to cumulative arrays
            intensities.append(intensity)
            energies.append(frame.energy)
        # Combine into a series
        series = pd.Series(intensities, index=energies)
        return series

    def plot_xanes_spectrum(self, ax=None):
        spectrum = self.xanes_spectrum()
        if ax is None:
            ax = new_axes()
        ax.plot(spectrum, marker='o', linestyle=":")
        ax.set_xlabel('Energy /eV')
        ax.set_ylabel('Overall absorbance')
        return ax

    def edge_jump_filter(self):
        """Calculate an image mask filter that represents the difference in
        signal across the X-ray edge."""
        # Sort frames into pre-edge and post-edge
        pre_edge = self.edge.pre_edge
        post_edge = self.edge.post_edge
        pre_images = []
        post_images = []
        for frame in self:
            if pre_edge[0] <= frame.energy <= pre_edge[1]:
                pre_images.append(frame.image_data)
            elif post_edge[0] <= frame.energy <= post_edge[1]:
                post_images.append(frame.image_data)
        # Convert lists to numpy arrays
        pre_images = np.array(pre_images)
        post_images = np.array(post_images)
        # Find average frames pre-edge/post-edge values
        pre_average = np.mean(pre_images, axis=0)
        post_average = np.mean(post_images, axis=0)
        # Subtract pre-edge from post-edge
        filtered_img = post_average - pre_average
        # Apply normalizer? (maybe later)
        return filtered_img

    def calculate_map(self):
        """Generate a map based on pixel-wise Xanes spectra. Default is to
        compute X-ray whiteline position."""
        # Get map data
        map_data = self.whiteline_map()
        # Create dataset
        name = self.hdf_group().name + "_map"
        try:
            del self.hdf_file()[name]
        except KeyError:
            pass
        self.hdf_file().create_dataset(name, data=map_data)
        self.map_name = name
        return self.hdf_file()[name]

    def masked_map(self):
        """Generate a map based on pixel-wise Xanes spectra and apply an
        edge-jump filter mask. Default is to compute X-ray whiteline
        position.
        """
        # Check for cached map of the whiteline position for each pixel
        if not self.map_name:
            map_data = self.calculate_map()
        else:
            map_data = self.hdf_file()[self.map_name]
        # Apply edge jump mask
        edge_jump = self.edge_jump_filter()
        threshold = filters.threshold_otsu(edge_jump)
        mask = edge_jump > threshold
        mask = morphology.dilation(mask)
        mask = np.logical_not(mask)
        masked_map = np.ma.array(map_data, mask=mask)
        return masked_map

    def plot_map(self, ax=None, norm_range=(None, None)):
        if ax is None:
            ax = new_axes()
        norm = Normalize(vmin=norm_range[0], vmax=norm_range[1])
        # Plot average absorbance
        # self[-1].plot_image(ax=ax)
        ax.imshow(self.masked_map(), cmap=self.cmap, norm=norm)
        return ax

    def plot_histogram(self, ax=None, norm_range=(None, None)):
        if ax is None:
            ax = new_axes()
        # Set normalizer
        norm = Normalize(norm_range[0], norm_range[1])
        masked_map = self.masked_map()
        n, bins, patches = ax.hist(masked_map[~masked_map.mask],
                                   bins=self.edge.energies())
        # Set colors on histogram
        for patch in patches:
            x_position = patch.get_x()
            cmap = cmaps[self.cmap]
            color = cmap(norm(x_position))
            patch.set_color(color)
        # Set axes decorations
        ax.set_xlabel("Whiteline position /eV")
        ax.set_ylabel("Pixels")
        ax.set_xlim(norm_range[0], norm_range[1])
        return ax

    def whiteline_map(self):
        """Calculate a map where each pixel is the energy of the whiteline."""
        print('Calculating whiteline map...', end='')
        # Determine indices of max frame per pixel
        imagestack = np.array([frame.image_data for frame in self])
        indices = np.argmax(imagestack, axis=0)
        # Map indices to energies
        map_energy = np.vectorize(lambda idx: self[idx].energy,
                                  otypes=[np.float])
        energies = map_energy(indices)
        print('done')
        return energies

    def hdf_file(self):
        # Determine filename
        try:
            file = h5py.File(self.hdf_filename, 'r+')
        except OSError as e:
            # HDF File does not exist, make a new one
            print('Creating new HDF5 file: {}'.format(self.hdf_filename))
            file = h5py.File(self.hdf_filename, 'w-')
            file.create_group(self.parent_groupname)
        return file

    def hdf_group(self):
        return self.hdf_file()[self.parent_groupname]

    def background_group(self):
        return self.hdf_file()[self.background_groupname]

    def hdf_node(self):
        """For use with HDFAttribute descriptor."""
        return self.hdf_group()

    def active_group(self):
        parent_group = self.hdf_group()
        if self.active_groupname is None:
            group = parent_group
        else:
            if not self.active_groupname in parent_group.keys():
                # Create group if necessary
                parent_group.create_group(self.active_groupname)
            group = parent_group[self.active_groupname]
        return group

    def add_frame(self, frame):
        setname_template = "{energy}_eV{serial}"
        frames_group = self.active_group()
        # Find a unique frame dataset name
        setname = setname_template.format(
            energy=frame.approximate_energy,
            serial=""
        )
        counter = 0
        while setname in frames_group.keys():
            counter += 1
            setname = setname_template.format(
                energy=frame.approximate_energy,
                serial="-" + str(counter)
            )
        # Name found, so create the actual dataset
        new_dataset = frame.create_dataset(setname=setname,
                                           hdf_group=frames_group)
        return setname

    def normalizer(self):
        # Find global limits
        global_min = 0
        global_max = 99999999999
        for frame in self:
            data = frame.image_data.value
            local_min = np.min(data)
            if local_min < global_min:
                global_min = local_min
            local_max = np.max(data)
            if local_max < global_max:
                global_max = local_max
        return Normalize(global_min, global_max)

    def gtk_viewer(self):
        viewer = GtkTxmViewer(frameset=self)
        viewer.show()


## Multiprocessing modules
