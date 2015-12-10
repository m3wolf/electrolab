from collections import namedtuple, OrderedDict
import os
import math
import warnings

import numpy as np
from matplotlib import pyplot
from scipy import ndimage
from skimage import img_as_float
from skimage.morphology import (closing, remove_small_objects, square, disk,
                                watershed, dilation)
from skimage.exposure import rescale_intensity
from skimage.measure import regionprops, label
from skimage.filters import threshold_otsu, threshold_adaptive, rank, threshold_li
from skimage.feature import peak_local_max
from skimage.segmentation import clear_border
from matplotlib.colors import Normalize

import plots
from hdf import HDFAttribute
from utilities import xycoord
from .particle import Particle

position = namedtuple('position', ('x', 'y', 'z'))

def average_frames(*frames):
    """Accept several frames and return the first frame with new image
    data. Assumes metadata from first frame in list."""
    new_image = np.zeros_like(frames[0].image_data, dtype=np.float)
    # Sum all images
    for frame in frames:
        new_image += frame.image_data
    # Divide to get average
    count = len(frames)
    new_image = new_image/len(frames)
    # Return average data as a txm frame
    new_frame = frames[0]
    new_frame.image_data = new_image
    return new_frame


class TXMFrame():
    """A single microscopy image at a certain energy."""

    image_data = np.zeros(shape=(1024,1024))
    _attrs = {}
    energy = HDFAttribute('energy', default=0.0)
    approximate_energy = HDFAttribute('approximate_energy', default=0.0)
    original_filename = HDFAttribute('original_filename', default=None)
    sample_position = HDFAttribute('sample_position',
                                   default=position(0, 0, 0),
                                   wrapper=lambda coords: position(*coords))
    approximate_position = HDFAttribute('approximate_position',
                                        default=position(0, 0, 0),
                                        wrapper=lambda coords: position(*coords))
    is_background = HDFAttribute('is_background', default=False)
    particle_labels_path = HDFAttribute('particle_labels_path', default=None)

    def __init__(self, file=None, frameset=None):
        if file:
            self.energy = file.energy()
            self.approximate_energy = round(self.energy, 1)
            self.original_filename = os.path.basename(file.filename)
            self.image_data = file.image_data()
            self.sample_position = file.sample_position()
            self.sample_name = file.sample_name
            self.position_name = file.position_name
            # self.reference_path = file.reference_path
            self.approximate_position = position(
                round(self.sample_position.x, -1),
                round(self.sample_position.y, -1),
                round(self.sample_position.z, -1)
            )
            self.is_background = file.is_background

    def __repr__(self):
        name = "<TXMFrame: {energy} eV at ({x}, {y}, {z})"
        return name.format(
            energy=int(self.energy),
            x=self.approximate_position.x,
            y=self.approximate_position.y,
            z=self.approximate_position.z,
        )

    def transmission_data(self, background_group):
        bg_data = self.background_dataset(group=background_group)
        return bg_data/np.exp(self.image_data)

    def background_dataset(self, group):
        key = self.image_data.name.split('/')[-1]
        return group[key]

    def hdf_node(self):
        return self.image_data

    def plot_image(self, data=None, ax=None, show_particles=True, *args, **kwargs):
        """Plot a frame's data image. Use frame.image_data if no data are
        given."""
        if ax is None:
            ax=plots.new_axes()
        if data is None:
            data = self.image_data
        # Determine physical dimensions for axes values
        y_pixels, x_pixels = data.shape
        um_per_pixel = self.um_per_pixel()
        center = self.sample_position
        left = center.x - x_pixels * um_per_pixel.x / 2
        right = center.x + x_pixels * um_per_pixel.x / 2
        bottom = center.y - y_pixels * um_per_pixel.y / 2
        top = center.y + y_pixels * um_per_pixel.y / 2
        extent = [left, right, bottom, top]
        im_ax = ax.imshow(data, *args, cmap='gray', extent=extent, **kwargs)
        # Plot particles
        if show_particles:
            self.plot_particle_labels(ax=im_ax.axes, extent=extent)
        # Set labels, etc
        ax.set_xlabel('µm')
        ax.set_ylabel('µm')
        im_ax.set_extent(extent)
        return ax

    def plot_particle_labels(self, ax=None, *args, **kwargs):
        """Plot the identified particles (as an overlay if ax is given)."""
        if ax is None:
            opacity = 1
            ax = plots.new_axes()
        else:
            opacity = 0.3
        if self.particle_labels_path:
            data = self.particle_labels()
            x = [particle.sample_position().x for particle in self.particles()]
            y = [particle.sample_position().y for particle in self.particles()]
            im_ax = ax.imshow(data, *args, alpha=opacity, **kwargs)
            ax.plot(x, y, linestyle="None", marker='o')
            ret = im_ax
        else:
            ret = None
        return ret

    def crop(self, top, left, bottom, right):
        """Reduce the image size to given box (in pixels)."""
        labels = self.particle_labels()
        # Move particle and labels to top left
        self.shift_data(x_offset=-left, y_offset=-top)
        self.shift_data(x_offset=-left, y_offset=-top,
                        dataset=self.particle_labels())
        # Shrink images to bounding box size
        self.image_data.resize((bottom-top, right-left))
        labels.resize((bottom-top, right-left))

    def shift_data(self, x_offset, y_offset, dataset=None):
        """Move the image within the view field by the given offsets in pixels.
        New values are rolled around to the other side.

        Arguments
        ---------
        x_offset : int
            Distance to move in pixels in x-diraction
        y_offset : int
            Distance to move in pixels in y-diraction
        dataset : Dataset
            Optional dataset to manipulate . If None, self.image_data will
            be used (default None)

        """
        if dataset is None:
            dataset = self.image_data
        # Roll along x axis
        new_data = np.roll(dataset, x_offset, axis=1)
        # Roll along y axis
        new_data = np.roll(new_data, y_offset, axis=0)
        # Commit shift image
        dataset.write_direct(new_data)
        # Update stored position information
        um_per_pixel = self.um_per_pixel()
        new_position = position(
            x=self.sample_position.x + y_offset * um_per_pixel.x,
            y=self.sample_position.y + y_offset * um_per_pixel.y,
            z=self.sample_position.z
        )
        self.sample_position = new_position
        return dataset.value

    def rebin(self, shape=None, factor=None):
        """Resample image into new shape. One of the kwargs `shape` or
        `factor` is required. Process is most effective when factors
        are powers of 2 (2, 4, 8, 16, etc).

        Kwargs:
        -------
        shape (tuple): The target shape for the new array. Will
            override `factor` if both are provided.
        factor (int): Factor by which to decrease the frame size. factor=2 would
            take a (1024, 1024) to (512, 512)

        """
        original_shape = self.image_data.shape
        if shape is None and factor is None:
            # Raise an error if not arguments are passed.
            raise ValueError("Must pass one of `shape` or `factor`")
        elif shape is None:
            # Determine new shape from factor if not provided.
            new_shape = tuple(int(dim/factor) for dim in original_shape)
        else:
            new_shape = shape
        # Check that the new shape is not larger than the old shape
        for idx, dim in enumerate(new_shape):
            if dim > original_shape[idx]:
                msg = 'New shape {new} is larger than original shape {original}.'
                msg = msg.format(new=new_shape, original=original_shape)
                raise ValueError(msg)
        data = self.image_data.value
        # Determine new dimensions
        sh = (new_shape[0],
              data.shape[0]//new_shape[0],
              new_shape[1],
              data.shape[1]//new_shape[1])
        new_data = data.reshape(sh).mean(-1).mean(1)
        # Resize existing dataset
        self.image_data.resize(new_shape)
        self.image_data.write_direct(new_data)
        return new_data

    def um_per_pixel(self):
        """Use image size and nominal image field-of view of 40µm x 40µm to
        compute spatial resolution."""
        um_per_pixel_x = 40/self.image_data.shape[1]
        um_per_pixel_y = 40/self.image_data.shape[0]
        return xycoord(x=um_per_pixel_x,
                       y=um_per_pixel_y)

    def create_dataset(self, setname, hdf_group):
        """Save data and metadata to an HDF dataset."""
        attrs = getattr(self.image_data, 'attrs', self._attrs)
        self.image_data = hdf_group.create_dataset(name=setname,
                                                   data=self.image_data,
                                                   maxshape=self.image_data.shape,
                                                   compression="gzip")
        # Set metadata attributes
        for attr_name in attrs.keys():
            self.image_data.attrs[attr_name] = attrs[attr_name]

    @classmethod
    def load_from_dataset(Cls, dataset):
        """Accept an HDF5 frame dataset and return a new frame."""
        new_frame = Cls()
        new_frame.image_data = dataset
        return new_frame

    def particle_labels(self):
        if self.particle_labels_path is None:
            res = calculate_particle_labels(self.image_data.value)
        else:
            res = self.image_data.file[self.particle_labels_path]
        return res

    def particles(self):
        labels = self.particle_labels()
        props = regionprops(labels, intensity_image=self.image_data)
        particles = []
        for prop in props:
            particles.append(Particle(regionprops=prop, frame=self))
        return particles

def calculate_particle_labels(data, return_intermediates=False,
                              min_distance=20):
    """Identify and label material particles in the image data.

    Generate and save a scikit-image style labels frame
    identifying the different particles. `return_all=True` returns
    a list of intermediates images (dict) instead of just the
    final result. Returns the final computed labels image, or a
    dictionary of all images (see kwarg return_intermediates.

    Parameters
    ----------
    return_intermediates : bool
        Return intermediate images as a dict (default False)
    min_distance : int
        How far away in pixels particle centers need to be in
        order to register as different particles (default 25)

    """
    # Shift image into range -1 to 1
    # normalizer = Normalize(vmin=self.image_data.value.min(),
    #                        vmax=self.image_data.value.max())
    original = data
    # equalized = skimage.exposure.equalize_hist(self.image_data.value)
    # Contrast stretching
    in_range = (data.min(), data.max())
    rescaled = rescale_intensity(original, in_range=in_range, out_range=(0, 1))
    equalized = rescaled
    # Stretch out the contrast to make identification easier
    # with warnings.catch_warnings():
    #     # Raises a lot of precision loss warnings that are irrelevant
    #     warnings.simplefilter("ignore")
    #     equalized = skimage.exposure.equalize_adapthist(rescaled, clip_limit=0.05)
    # Identify foreground vs background with Otsu filter
    # threshold = threshold_otsu(equalized)
    threshold = threshold_li(equalized)
    mask = equalized > threshold
    # Fill in the shapes a little
    closed = dilation(mask, square(5))
    # Remove features at the edge of the frame since they can be incomplete
    border_cleared = clear_border(closed)
    # Discard small particles
    # Determine minimum size for discarding objects
    average_shape = sum(border_cleared.shape)/len(border_cleared.shape)
    min_size = 8. * average_shape
    large_only = remove_small_objects(border_cleared, min_size=min_size)
    # Fill in the shapes a lot to round them out
    reclosed = closing(large_only, disk(20))
    # Expand the particles to make sure we capture the edges
    dilated = dilation(reclosed, disk(10))
    # Compute each pixel's distance from the edge of a blob
    distances = ndimage.distance_transform_edt(dilated)
    in_range = (distances.min(), distances.max())
    distances = rescale_intensity(distances, in_range=in_range, out_range=(0, 1))
    # Blur the distances to help avoid split particles
    mean_distances = rank.mean(distances, disk(5))
    # Use the local distance maxima as peak centers and compute labels
    local_maxima = peak_local_max(
        mean_distances,
        indices=False,
        min_distance=min_distance,
        # footprint=np.ones((64, 64)),
        labels=dilated
    )
    markers = label(local_maxima)
    labels = watershed(-mean_distances, markers, mask=dilated)
    if return_intermediates:
        result = OrderedDict()
        result['original'] = original
        result['equalized'] = equalized
        result['mask'] = mask
        result['closed'] = closed
        result['border_cleared'] = border_cleared
        result['large_only'] = large_only
        result['reclosed'] = reclosed
        result['dilated'] = dilated
        result['mean_distances'] = mean_distances
        result['distances'] = distances
        result['local_maxima'] = local_maxima
        result['markers'] = markers
        result['labels'] = labels
    else:
        result = labels
    return result
