import functools
import os
from collections import defaultdict
import re
import math
import multiprocessing
import queue
import warnings

import pandas as pd
from matplotlib import pyplot, cm
from matplotlib.colors import Normalize, BoundaryNorm
from mapping.colormaps import cmaps
import h5py
import numpy as np
from skimage import morphology, filters, feature, transform, restoration, exposure

from utilities import prog, xycoord
from .frame import (
    TXMFrame, average_frames, calculate_particle_labels, pixel_to_xy,
    apply_reference, position, Pixel)
from .gtk_viewer import GtkTxmViewer
from .plotter import FramesetPlotter, FramesetMoviePlotter
from plots import new_axes, new_image_axes, DegreeFormatter, ElectronVoltFormatter
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
    active_groupname = None
    latest_groupname = HDFAttribute('latest_groupname')
    background_groupname = HDFAttribute('background_groupname')
    active_particle_idx = HDFAttribute('active_particle_idx', default=None,
                                       group_func='active_group')
    latest_labels = HDFAttribute('latest_labels', default='particle_labels')
    map_name = HDFAttribute('map_name', group_func='active_group')
    cmap = 'plasma'
    def __init__(self, filename, groupname, edge=None):
        self.hdf_filename = filename
        self.parent_groupname = groupname
        self.edge = edge
        # Check to make sure a valid group is given
        if filename:
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
            keys = list(self.active_group().keys())
            name = keys[index]
            frame = TXMFrame.load_from_dataset(self.active_group()[name])
        return frame

    def clear_caches(self):
        """Clear cached function values so they will be recomputed with fresh
        data"""
        self.xanes_spectrum.cache_clear()
        self.image_normalizer.cache_clear()
        self.edge_jump_filter.cache_clear()

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
        """Set the frameset to retrieve image data from a different hdf
        group. Special value 'background_frames' sets to the reference
        image used during importing.
        """
        valid_groups = list(self.hdf_group().keys()) + ['background_frames']
        if name not in valid_groups:
            msg = "{name} is not a valid group. Choices are {choices}."
            raise exceptions.GroupKeyError(
                msg.format(name=name, choices=valid_groups)
            )
        if name == 'background_frames':
            # Clear cached value
            self.active_groupname = self.background_groupname
        else:
            self.active_groupname = name
        self.clear_caches()

    def fork_group(self, name):
        """Create a new, copy of the current active group inside the HDF
        parent with name: `name`.
        """
        # Create an empty group
        try:
            del self.hdf_group()[name]
        except KeyError as e:
            # Ignore error only if group doesn't exists
            if not e.args[0] == "Couldn't delete link (Can't delete self)":
                raise
        # Copy the old data
        self.hdf_group().copy(source=self.active_groupname, dest=name)
        # Update the group name
        self.latest_groupname = name
        self.switch_group(name)
        # Delete reference to the old map (rather recalculate than risk a stale map)
        try:
            del self.map_name
        except KeyError:
            pass

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

    def apply_references(self, bg_groupname):
        """Apply reference corrections for this frameset. Converts raw
        intensity frames to absorbance frames."""
        self.background_groupname = bg_groupname
        self.fork_group('absorbance_frames')
        bg_group = self.hdf_file()[bg_groupname]
        for frame in prog(self, "Reference correction"):
            key = frame.image_data.name.split('/')[-1]
            bg_dataset = bg_group[key]
            new_data = apply_reference(frame.image_data.value,
                                       reference_data=bg_dataset.value)
            # Resize the dataset if necessary
            if new_data.shape != frame.image_data.shape:
                frame.image_data.resize(new_data.shape)
            frame.image_data.write_direct(new_data)
        # self.hdf_file().close()

    def correct_magnification(self):
        """Correct for changes in magnification at different energies.

        As the X-ray energy increases, the focal length of the zone
        plate changes and so the image is zoomed-out at higher
        energies. This method applies a correction to each frame to
        make the magnification similar to that of the first frame.
        """
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

    def align_frames(self, new_name="aligned_frames", reference_frame=None):
        """Use phase correlation algorithm to line up the frames. All frames
        have their sample position set set to (0, 0) since we don't
        know which one is the real position.
        """
        # Guess best reference frame
        if reference_frame is None:
            spectrum = self.xanes_spectrum()
            reference_frame = np.argmax(spectrum.values)
        # Create new data groups to hold shifted image data
        self.fork_group(new_name)
        self.fork_labels(new_name + "_labels")
        reference_image = self[reference_frame].image_data.value
        # Multiprocessing setup
        def worker(payload):
            key, data, labels = payload
            # Determine what the new translation should be
            results = feature.register_translation(reference_image, data,
                                                   upsample_factor=20)
            shift, error, diffphase = results
            shift = xycoord(-shift[1], -shift[0])
            # Apply net transformation with bicubic interpolation
            transformation = transform.SimilarityTransform(translation=shift)
            new_data = transform.warp(data, transformation,
                                      order=3, mode="wrap")
            # Transform labels
            original_dtype = labels.dtype
            labels = labels.astype(np.float64)
            new_labels = transform.warp(labels, transformation, order=0, mode="constant")
            new_labels = new_labels.astype(original_dtype)
            return (key, new_data, new_labels)
        def process_result(payload):
            key, data, labels = payload
            frame = self[key]
            frame.image_data.write_direct(data)
            frame.particle_labels().write_direct(labels)
        description = "Aligning to frame [{}]".format(reference_frame)
        queue = smp.Queue(worker=worker, result_callback=process_result,
                          totalsize=len(self), description=description)
        for frame in self:
            key = frame.key()
            # Prepare data arrays
            data = frame.image_data.value
            labels = frame.particle_labels().value
            # Launch transformation for this frame
            queue.put((key, data, labels))
        queue.join()
        # Update new positions
        for frame in self:
            frame.sample_position = position(0, 0, frame.sample_position.z)

    def align_to_particle(self, loc, new_name, reference_frame=None):
        """Use template matching algorithm to line up the frames. Similar to
        `align_frames` but matches only to the particle closest to the
        argument `loc`.
        """
        # Autoguess best reference frame
        if reference_frame is None:
            spectrum = self.xanes_spectrum()
            reference_frame = np.argmax(spectrum.values)
        # Create new data groups to hold shifted image data
        self.fork_group(new_name)
        self.fork_labels(new_name + "_labels")
        # Determine which particle to use
        particle = self[reference_frame].activate_closest_particle(loc=loc)
        bbox = particle.bbox()
        # reference_img = particle.masked_frame_image()
        # particle_img = np.ma.array(particle.image(), mask=particle.mask())
        particle_img = np.copy(particle.image())
        # Set all values outside the particle itself to 0
        particle_img[np.logical_not(particle.mask())] = 0
        reference_key = self[reference_frame].key()
        reference_img = self[reference_frame].image_data.value
        reference_match = feature.match_template(reference_img, particle_img, pad_input=True)
        reference_center = np.unravel_index(reference_match.argmax(),
                                            reference_match.shape)
        reference_center = Pixel(vertical=reference_center[0],
                                 horizontal=reference_center[1])
        # Multiprocessing setup
        def worker(payload):
            key, data, labels = payload
            # Determine where the reference particle is in this frame's image
            match = feature.match_template(data, particle_img, pad_input=True)
            center = np.unravel_index(match.argmax(), match.shape)
            center = Pixel(vertical=center[0], horizontal=center[1])
            # Determine the net translation necessary to align to reference frame
            shift = [
                center.horizontal - reference_center.horizontal,
                center.vertical - reference_center.vertical,
            ]
            if key == reference_key:
                # Sanity check to ensure that reference frame does not shift
                assert shift == [0, 0], "Reference frame is shifted by " + shift
                ret = (key, data, labels)
            else:
                # Apply the translation with bicubic interpolation
                transformation = transform.SimilarityTransform(translation=shift)
                new_data = transform.warp(data, transformation,
                                          order=3, mode="wrap")
                # Transform labels
                original_dtype = labels.dtype
                labels = labels.astype(np.float64)
                new_labels = transform.warp(labels, transformation, order=0, mode="constant")
                new_labels = new_labels.astype(original_dtype)
                ret = (key, new_data, new_labels)
            return ret
        def process_result(payload):
            key, data, labels = payload
            frame = self[key]
            frame.image_data.write_direct(data)
            frame.particle_labels().write_direct(labels)
            frame.activate_closest_particle(loc=loc)
        description = "Aligning to frame [{}]".format(reference_frame)
        queue = smp.Queue(worker=worker, result_callback=process_result,
                          totalsize=len(self), description=description)
        for frame in self:
            key = frame.key()
            # Prepare data arrays
            data = frame.image_data.value
            labels = frame.particle_labels().value
            # Launch transformation for this frame
            queue.put((key, data, labels))
        queue.join()
        # Update new positions
        for frame in self:
            frame.sample_position = position(0, 0, frame.sample_position.z)
        return reference_match

    def crop_to_particle(self, new_name='cropped_particle', loc=None):
        """Reduce the image size to just show the particle in question."""
        # Create new HDF5 groups
        self.fork_group(new_name)
        self.fork_labels('cropped_labels')
        # Activate particle if necessary
        if loc is not None:
            for frame in prog(self, 'Idetifying closest particle'):
                particle = frame.activate_closest_particle(loc=loc)
        # Make sure an active particle is assigned to all frames
        for frame in self:
            if frame.active_particle_idx is None:
                msg = "Frame {} has no particle assigned".format(frame)
                raise exceptions.NoParticleError(msg)
        # Determine largest bounding box based on all energies
        boxes = [frame.particles()[frame.active_particle_idx].bbox()
                 for frame in self]
        left = min([box.left for box in boxes])
        top = min([box.top for box in boxes])
        bottom = max([box.bottom for box in boxes])
        right = max([box.right for box in boxes])
        # Make sure the expanded box is square
        def expand_dims(lower, upper, target):
            center = (lower+upper)/2
            new_lower = center - target/2
            new_upper = center + target/2
            return (new_lower, new_upper)
        if right-left > bottom-top:
            top, bottom = expand_dims(top, bottom, target=right-left)
        elif  bottom-top > right-left:
            left, right = expand_dims(left, right, target=bottom-top)
        assert abs(left-right) == abs(bottom-top)
        # Roll each image to have the particle top left
        for frame in prog(self, 'Cropping frames'):
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
        for frame in prog(self, 'Computing true center'):
            n+=1
            total_x += frame.sample_position.x
            total_y += frame.sample_position.y
        global_x = total_x / n
        global_y = total_y / n
        for frame in prog(self, 'Aligning frames'):
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
        labels_groupname = self.active_groupname + "_labels"
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
        self.fork_labels('rebinned_labels')
        for frame in prog(self, "Rebinning"):
            frame.rebin(shape=shape, factor=factor)

    def plot_mean_image(self, ax=None):
        if ax is None:
            ax = new_image_axes()
        data = self.mean_image()
        artist = ax.imshow(data, extent=self.extent(), cmap='gray')
        return artist

    def mean_image(self):
        """Determine an overall image by taken the mean intensity of each
        pixel across all frames."""
        frames = np.array([f.image_data for f in self])
        avg_frame = np.mean(frames, axis=0)
        return avg_frame

    @functools.lru_cache()
    def xanes_spectrum(self, pixel=None):
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
                masked_data = np.copy(data)
                masked_data[mask] = 0
            else:
                masked_data = data
            if pixel is None:
                # Sum absorbances for datasets
                intensity = np.sum(masked_data)/np.prod(masked_data.shape)
            else:
                # Calculate intensity just for one (unmasked) pixel
                intensity = data[pixel.vertical][pixel.horizontal]
                # intensity = data[pixel.vertical][pixel.horizontal]
            # Add to cumulative arrays
            intensities.append(intensity)
            energies.append(frame.energy)
        # Combine into a series
        series = pd.Series(intensities, index=energies)
        return series

    def plot_xanes_spectrum(self, ax=None, pixel=None, norm_range=None):
        if norm_range is None:
            norm_range = (self.edge.map_range[0], self.edge.map_range[1])
            norm = Normalize(*norm_range)
        spectrum = self.xanes_spectrum(pixel=pixel)
        if ax is None:
            ax = new_axes()
        # Color code the markers by energy
        colors = []
        for energy in spectrum.index:
            cmap = cm.get_cmap(self.cmap)
            colors.append(cmap(norm(energy)))
        ax.plot(spectrum, linestyle=":")
        xlim = ax.get_xlim(); ylim = ax.get_ylim()
        scatter = ax.scatter(spectrum.index, spectrum.values, c=colors, s=25)
        # Restore axes limits, they get messed up by scatter()
        ax.set_xlim(*xlim); ax.set_ylim(*ylim)
        ax.set_xlabel('Energy /eV')
        ax.set_ylabel('Absorbance')
        if pixel is not None:
            xy = pixel_to_xy(pixel, extent=self.extent(), shape=self.map_shape())
            title = 'XANES Spectrum at ({x}, {y}) = {val}'
            title = title.format(x=round(xy.x, 2),
                                 y=round(xy.y, 2),
                                 val=self.masked_map()[pixel.vertical][pixel.horizontal])
            ax.set_title(title)
        # Plot lines at edge of normalization range
        ax.axvline(x=norm_range[0], linestyle='-', color="0.55", alpha=0.4)
        ax.axvline(x=norm_range[1], linestyle='-', color="0.55", alpha=0.4)
        return scatter

    def plot_edge_jump(self, ax=None, alpha=1):
        """Plot the results of the edge jump filter."""
        if ax is None:
            ax = new_image_axes()
        ej = self.edge_jump_filter()
        artist = ax.imshow(ej, extent=self.extent(), cmap=self.cmap, alpha=alpha)
        ax.set_xlabel('µm')
        ax.set_ylabel('µm')
        return artist

    @functools.lru_cache()
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

    def calculate_map(self, new_name=None):
        """Generate a map based on pixel-wise Xanes spectra. Default is to
        compute X-ray whiteline position."""
        # Get default hdf group name
        if new_name is None:
            new_name = "{parent}/{group}_map".format(parent=self.parent_groupname,
                                                     group=self.active_groupname)
        # Get map data
        map_data = self.whiteline_map()
        # Create dataset
        try:
            del self.hdf_file()[new_name]
        except KeyError:
            pass
        self.hdf_file().create_dataset(new_name, data=map_data)
        self.map_name = new_name
        return self.hdf_file()[new_name]

    def masked_map(self, edge_jump_filter=True):
        """Generate a map based on pixel-wise Xanes spectra and apply an
        edge-jump filter mask. Default is to compute X-ray whiteline
        position.
        """
        # Check for cached map of the whiteline position for each pixel
        if not self.map_name:
            map_data = self.calculate_map()
        else:
            map_data = self.hdf_file()[self.map_name]
        if edge_jump_filter:
            # Apply edge jump mask
            edge_jump = self.edge_jump_filter()
            threshold = filters.threshold_otsu(edge_jump)
            mask = edge_jump > 0.5 * threshold
            mask = morphology.dilation(mask)
            mask = np.logical_not(mask)
        else:
            # No-op filter
            mask = np.zeros_like(map_data)
        masked_map = np.ma.array(map_data, mask=mask)
        return masked_map

    def map_shape(self):
        return self[0].image_data.shape

    def extent(self):
        """Determine physical dimensions for axes values."""
        return self[0].extent()

    def plot_map(self, plotter=None, ax=None, norm_range=None, alpha=1,
                 edge_jump_filter=False, return_type="axes"):
        """Use a default frameset plotter to draw a map of the chemical data."""
        if plotter is None:
            plotter = FramesetPlotter(frameset=self, map_ax=ax)
        plotter.draw_map(norm_range=norm_range, alpha=alpha,
                         edge_jump_filter=edge_jump_filter)
        return plotter

    def plot_histogram(self, ax=None, norm_range=None):
        if ax is None:
            ax = new_axes()
        # Set normalizer
        if norm_range is None:
            norm = Normalize(self.edge.map_range[0], self.edge.map_range[1])
        else:
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
        ax.set_xlim(norm.vmin, norm.vmax)
        ax.xaxis.get_major_formatter().set_useOffset(False)
        return ax

    def save_movie(self, filename, *args, **kwargs):
        """Save an animation of all the frames and XANES to the specified
        filename."""
        pl = FramesetMoviePlotter(frameset=self)
        pl.create_axes()
        pl.connect_animation()
        pl.save_movie(filename=filename, *args, **kwargs)

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
        if self.hdf_filename is not None:
            # Determine filename
            try:
                file = h5py.File(self.hdf_filename, 'r+')
            except OSError as e:
                print(e)
                # HDF File does not exist, make a new one
                print('Creating new HDF5 file: {}'.format(self.hdf_filename))
                file = h5py.File(self.hdf_filename, 'w-')
                file.create_group(self.parent_groupname)
        else:
            file = None
        return file

    def hdf_group(self):
        return self.hdf_file()[self.parent_groupname]

    def background_group(self):
        return self.hdf_file()[self.background_groupname]

    def hdf_node(self):
        """For use with HDFAttribute descriptor."""
        return self.hdf_group()

    def is_background(self):
        return self.active_groupname == self.background_groupname

    def active_group(self):
        parent_group = self.hdf_group()
        if self.active_groupname is None:
            group = parent_group
        elif self.is_background():
            # Background frames are relative to file root
            group = self.hdf_file()[self.active_groupname]
        else:
            # Create group if necessary
            if self.active_groupname not in parent_group.keys():
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
        # counter = 0
        # while setname in frames_group.keys():
        #     counter += 1
        #     setname = setname_template.format(
        #         energy=frame.approximate_energy,
        #         serial="-" + str(counter)
        #     )
        # Name found, so create the actual dataset
        new_dataset = frame.create_dataset(setname=setname,
                                           hdf_group=frames_group)
        return setname

    def background_normalizer(self):
        # Find global limits
        global_min = 0
        global_max = 99999999999
        bg_group = self.background_group()
        for key in bg_group.keys():
            data = bg_group[key].value
            local_min = np.min(data)
            if local_min < global_min:
                global_min = local_min
            local_max = np.max(data)
            if local_max < global_max:
                global_max = local_max
        return Normalize(global_min, global_max)

    @functools.lru_cache()
    def image_normalizer(self):
        # Find global limits
        global_min = 99999999999
        global_max = 0
        for frame in self:
            data = frame.image_data.value
            local_min = np.min(data)
            if local_min < global_min:
                global_min = local_min
            local_max = np.max(data)
            if local_max > global_max:
                global_max = local_max
        return Normalize(global_min, global_max)

    def gtk_viewer(self):
        viewer = GtkTxmViewer(frameset=self)
        viewer.show()
