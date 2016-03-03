# -*- coding: utf-8 -*-
#
# Copyright © 2016 Mark Wolf
#
# This file is part of scimap.
#
# Scimap is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# Scimap is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with Scimap. If not, see <http://www.gnu.org/licenses/>.

from collections import namedtuple, OrderedDict
import os
import math
import warnings

import dateutil.parser
import numpy as np
from scipy import ndimage
from skimage.morphology import (disk, closing, remove_small_objects,
                                watershed, dilation)
from skimage.exposure import rescale_intensity
from skimage.measure import regionprops, label
from skimage.filters import threshold_adaptive, rank
from skimage.feature import peak_local_max

import exceptions
import plots
from hdf import HDFAttribute
from utilities import xycoord, Pixel, shape
from .particle import Particle

position = namedtuple('position', ('x', 'y', 'z'))
Extent = namedtuple('extent', ('left', 'right', 'bottom', 'top'))


def rebin_image(data, shape):
    """Resample image into new shape, but only if the new dimensions are
    smaller than the old. This is not meant to apply zoom corrections,
    only correct sizes in powers of two. Eg, a 2048x2048 images can be
    down-sampled to 1024x1024.

    Kwargs:
    -------
    shape (tuple): The target shape for the new array.

    """
    # Return original data is shapes are the same
    if data.shape == shape:
        return data
    # Check that the new shape is not larger than the old shape
    for idx, dim in enumerate(shape):
        if dim > data.shape[idx]:
            msg = 'New shape {new} is larger than original shape {original}.'
            msg = msg.format(new=shape, original=data.shape)
            raise ValueError(msg)
    # Determine new dimensions
    sh = (shape[0],
          data.shape[0] // shape[0],
          shape[1],
          data.shape[1] // shape[1])
    # new_data = data.reshape(sh).mean(-1).mean(1)
    new_data = data.reshape(sh).sum(-1).sum(1)
    return new_data


def apply_reference(data, reference_data):
    """Apply a reference correction to a raw image. This turns intensity
    data into absorbance data. If data and reference data have
    different shapes, they will be down-samples to the lower of the
    two.

    Arguments:
    ----------
    data (numpy array): The sample image to be corrected.

    reference_data (numpy array): The reference data that will be
        corrected against.
    """
    # Rebin datasets in case they don't match
    min_shape = [
        min([ds.shape[0] for ds in [data, reference_data]]),
        min([ds.shape[1] for ds in [data, reference_data]]),
    ]
    reference_data = rebin_image(reference_data, shape=min_shape)
    data = rebin_image(data, shape=min_shape)
    new_data = np.log10(reference_data / data)
    return new_data


def xy_to_pixel(xy, extent, shape):
    """Take an xy location on an image and convert it to a pixel location
    suitable for numpy indexing."""
    ratio_x = (xy.x - extent.left) / (extent.right - extent.left)
    pixel_h = int(round(ratio_x * shape[1]))
    ratio_y = (xy.y - extent.bottom) / (extent.top - extent.bottom)
    # (1 - ratio) for y because images are top indexed
    pixel_v = int(round((1 - ratio_y) * shape[0]))
    return Pixel(vertical=pixel_v, horizontal=pixel_h)


def pixel_to_xy(pixel, extent, shape):
    """Take an xy location on an image and convert it to a pixel location
    suitable for numpy indexing."""
    # ratio_x = (xy.x-extent.left)/(extent.right-extent.left)
    # pixel_h = int(round(ratio_x * shape[1]))
    # ratio_y = (xy.y-extent.bottom)/(extent.top-extent.bottom)
    # # (1 - ratio) for y because images are top indexed
    # pixel_v = int(round((1 - ratio_y) * shape[0]))
    ratio_h = (pixel.horizontal / shape[1])
    x = extent.left + ratio_h * (extent.right - extent.left)
    ratio_v = (pixel.vertical / shape[0])
    y = extent.top - ratio_v * (extent.top - extent.bottom)
    return xycoord(x=x, y=y)


def average_frames(*frames):
    """Accept several frames and return the first frame with new image
    data. Assumes metadata from first frame in list."""
    new_image = np.zeros_like(frames[0].image_data, dtype=np.float)
    # Sum all images
    for frame in frames:
        new_image += frame.image_data
    # Divide to get average
    new_image = new_image / len(frames)
    # Return average data as a txm frame
    new_frame = frames[0]
    new_frame.image_data = new_image
    return new_frame


class TXMFrame():
    """A single microscopy image at a certain energy."""

    image_data = np.zeros(shape=(1024, 1024))
    _attrs = {}
    energy = HDFAttribute('energy', default=0.0)
    approximate_energy = HDFAttribute('approximate_energy', default=0.0)
    original_filename = HDFAttribute('original_filename', default=None)
    _sample_position = HDFAttribute('sample_position',
                                    default=position(0, 0, 0),
                                    wrapper=lambda coords: position(*coords))
    approximate_position = HDFAttribute(
        'approximate_position',
        default=position(0, 0, 0),
        wrapper=lambda coords: position(*coords)
    )
    _starttime = HDFAttribute('starttime')
    _endtime = HDFAttribute('endtime')
    is_background = HDFAttribute('is_background', default=False)
    particle_labels_path = HDFAttribute('particle_labels_path', default=None)
    active_particle_idx = HDFAttribute('active_particle_idx', default=None)

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
        name = "<TXMFrame: {energy} eV at ({x}, {y}, {z})>"
        return name.format(
            energy=int(self.energy),
            x=self.approximate_position.x,
            y=self.approximate_position.y,
            z=self.approximate_position.z,
        )

    @property
    def sample_position(self):
        actual_position = self._sample_position
        return position(20, 20, actual_position.z)

    @sample_position.setter
    def sample_position(self, new_position):
        self._sample_position = new_position

    @property
    def starttime(self):
        return dateutil.parser.parse(self._starttime)

    @starttime.setter
    def starttime(self, newdt):
        self._starttime = newdt.isoformat()

    @property
    def endtime(self):
        return dateutil.parser.parse(self._endtime)

    @endtime.setter
    def endtime(self, newdt):
        self._endtime = newdt.isoformat()

    def transmission_data(self, background_group):
        bg_data = self.background_dataset(group=background_group)
        return bg_data / np.exp(self.image_data)

    def background_dataset(self, group):
        return group[self.key()]

    def key(self):
        return self.image_data.name.split('/')[-1]

    def hdf_node(self):
        return self.image_data

    def extent(self, shape=None):
        """Determine physical dimensions for axes values."""
        if shape is None:
            shape = self.image_data.shape
        y_pixels, x_pixels = shape
        um_per_pixel = self.um_per_pixel()
        center = self.sample_position
        left = center.x - x_pixels * um_per_pixel.x / 2
        right = center.x + x_pixels * um_per_pixel.x / 2
        bottom = center.y - y_pixels * um_per_pixel.y / 2
        top = center.y + y_pixels * um_per_pixel.y / 2
        return Extent(left=left, right=right, bottom=bottom, top=top)

    def plot_histogram(self, ax=None, bins=100, *args, **kwargs):
        if ax is None:
            ax = plots.new_axes()
        ax.set_xlabel('Absorbance (AU)')
        ax.set_ylabel('Occurences')
        data = np.nan_to_num(self.image_data.value)
        artist = ax.hist(data.flatten(), bins=bins, *args, **kwargs)
        return artist

    def plot_image(self, data=None, ax=None,
                   show_particles=True, *args, **kwargs):
        """Plot a frame's data image. Use frame.image_data if no data are
        given."""
        if ax is None:
            ax = plots.new_image_axes()
        if data is None:
            data = self.image_data
        extent = self.extent(shape=data.shape)
        im_ax = ax.imshow(data, *args, cmap='gray', extent=extent, **kwargs)
        # Plot particles
        if show_particles:
            self.plot_particle_labels(ax=im_ax.axes, extent=extent)
        # Set labels, etc
        ax.set_xlabel('µm')
        ax.set_ylabel('µm')
        im_ax.set_extent(extent)
        return im_ax

    def plot_particle_labels(self, ax=None, *args, **kwargs):
        """Plot the identified particles (as an overlay if ax is given)."""
        artists = []
        # Get axes extent (if passed) or use default
        extent = kwargs.pop('extent', self.extent())
        if ax is None:
            opacity = 1
            ax = plots.new_image_axes()
        else:
            opacity = 0.3
        if self.particle_labels_path:
            data = self.particle_labels()
            # Mask out anything not labeled (ie. set to zero)
            mask = np.logical_not(data.value.astype(np.bool))
            masked_data = np.ma.array(data, mask=mask)
            artists.append(ax.imshow(masked_data, *args, alpha=opacity,
                                     cmap="Dark2", extent=extent, **kwargs))
            # Plot text for label index
            particles = self.particles()
            xs = [particle.sample_position().x for particle in particles]
            ys = [particle.sample_position().y for particle in particles]
            for idx, x in enumerate(xs):
                y = ys[idx]
                txt = ax.text(x, y, str(idx),
                              horizontalalignment='center',
                              verticalalignment='center')
                artists.append(txt)
        return artists

    def remove_outliers(self, sigma):
        """Mark as invalid any pixels more that `sigma` standard deviations
        away from the median."""
        d = np.abs(self.image_data - np.median(self.image_data))
        sdev = np.std(self.image_data)
        # mdev = np.median(d)
        s = d / sdev if sdev else 0.
        self.image_data[s >= sigma] = 0

    def crop(self, top, left, bottom, right):
        """Reduce the image size to given box (in pixels)."""
        labels = self.particle_labels()
        # Move particle and labels to top left
        self.shift_data(x_offset=-left, y_offset=-top)
        self.shift_data(x_offset=-left, y_offset=-top,
                        dataset=self.particle_labels())
        # Shrink images to bounding box size
        new_shape = shape(rows=(bottom - top), columns=(right - left))
        self.image_data.resize(new_shape)
        labels.resize(new_shape)
        # Reassign the active particle index (assume largest particle)
        areas = [particle.area() for particle in self.particles()]
        new_idx = areas.index(max(areas))
        self.active_particle_idx = new_idx

    def shift_data(self, x_offset, y_offset, dataset=None):
        """Move the image within the view field by the given offsets in
        pixels.  New values are rolled around to the other side.

        Arguments
        ---------
        x_offset : int
            Distance to move in pixels in x-diraction
        y_offset : int
            Distance to move in pixels in y-diraction
        dataset : Dataset
            Optional dataset to manipulate . If None, self.image_data
            will be used (default None)
        """
        if dataset is None:
            dataset = self.image_data
        # Roll along x axis
        new_data = np.roll(dataset, int(x_offset), axis=1)
        # Roll along y axis
        new_data = np.roll(new_data, int(y_offset), axis=0)
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
        are powers of 2 (2, 4, 8, 16, etc). New shape is calculated
        and passed to the rebin_image() function.

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
            new_shape = tuple(int(dim / factor) for dim in original_shape)
        else:
            new_shape = shape
        # Calculate new, rebinned image data
        new_data = rebin_image(self.image_data.value, shape=new_shape)
        # Resize existing dataset
        self.image_data.resize(new_shape)
        self.image_data.write_direct(new_data)
        # Calculate new, rebinned particle labels
        try:
            labels = self.particle_labels()
        except exceptions.NoParticleError:
            pass
        else:
            new_labels = rebin_image(labels.value, shape=new_shape)
            # Resize existing particle labels
            labels.resize(new_shape)
            labels.write_direct(new_labels)
        return new_data

    def um_per_pixel(self):
        """Use image size and nominal image field-of view of 40µm x 40µm to
        compute spatial resolution."""
        um_per_pixel_x = 40 / self.image_data.shape[1]
        um_per_pixel_y = 40 / self.image_data.shape[0]
        return xycoord(x=um_per_pixel_x,
                       y=um_per_pixel_y)

    def create_dataset(self, setname, hdf_group):
        """Save data and metadata to an HDF dataset."""
        if setname in hdf_group.keys():
            msg = "{name} already exists in group {group}"
            msg = msg.format(name=setname, group=hdf_group.name)
            raise exceptions.DatasetExistsError(msg)
        else:
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
            raise exceptions.NoParticleError
        else:
            labels = self.image_data.file[self.particle_labels_path]
        return labels

    def activate_closest_particle(self, loc):
        """Get a particle that's closest to frame location and save for future
        reference."""
        if loc:  # Some calling routines may pass `None`
            self.active_particle_idx = self.closest_particle_idx(loc)
            return self.particles()[self.active_particle_idx]
        else:
            return None

    def closest_particle_idx(self, loc):
        """Get a particle that's closest to frame location."""
        particles = self.particles()
        current_min = 999999
        current_idx = None
        for idx, particle in enumerate(particles):
            center = particle.sample_position()
            distance = math.sqrt(
                (loc[0] - center[0])**2 + (loc[1] - center[1])**2
            )
            if distance < current_min:
                # New closest match
                current_min = distance
                current_idx = idx
        return current_idx

    def particles(self):
        labels = self.particle_labels()
        props = regionprops(labels, intensity_image=self.image_data)
        particles = []
        for prop in props:
            particles.append(Particle(regionprops=prop, frame=self))
        return particles


def calculate_particle_labels(data, return_intermediates=False,
                              min_distance=0.016):
    """Identify and label material particles in the image data.

    Generate and save a scikit-image style labels frame identifying
    the different particles. Returns an array with the same shape as
    `data` with values ap`oximating the particles in the frame. If
    kwarg `return_all` is truthy, returns an ordered dictionary of the
    intermediate arrays calculated during the algorithm (useful for
    debugging).

    Parameters
    ----------
    return_intermediates : bool
        Return intermediate images as a dict (default False)
    min_distance : float
        How far away (as a portion of image size) particle centers need to be in
        order to register as different particles (default 0.016)

    """
    with warnings.catch_warnings():
        # Supress precision loss warnings when we convert to binary mask
        warnings.simplefilter("ignore")
        # Shift image into range -1 to 1 by contrast stretching
        original = data
        in_range = (data.min(), data.max())
        equalized = rescale_intensity(original,
                                      in_range=in_range,
                                      out_range=(0, 1))
        # Identify foreground vs background with adaptive Otsu filter
        average_shape = sum(equalized.shape) / len(equalized.shape)
        # threshold = filters.threshold_otsu(equalized)
        # multiplier = 1
        block_size = average_shape / 2.72  # Determined emperically
        mask = threshold_adaptive(equalized, block_size=block_size, offset=0)
        mask_copy = np.copy(mask)
        # Determine minimum size for discarding small objects
        min_size = 4. * average_shape
        large_only = remove_small_objects(mask, min_size=min_size)
        # Fill in the shapes a lot to round them out
        reclosed = closing(large_only, disk(average_shape * 20 / 1024))
        # Expand the particles to make sure we capture the edges
        dilated = dilation(reclosed, disk(average_shape * 10 / 1024))
        # Compute each pixel's distance from the edge of a blob
        distances = ndimage.distance_transform_edt(dilated)
        in_range = (distances.min(), distances.max())
        distances = rescale_intensity(distances,
                                      in_range=in_range,
                                      out_range=(0, 1))
        # Blur the distances to help avoid split particles
        mean_distances = rank.mean(distances, disk(average_shape / 64))
        # Use the local distance maxima as peak centers and compute labels
        local_maxima = peak_local_max(
            mean_distances,
            indices=False,
            min_distance=min_distance * average_shape,
            labels=dilated
        )
        markers = label(local_maxima)
        labels = watershed(-mean_distances, markers, mask=dilated)
    if return_intermediates:
        result = OrderedDict()
        result['original'] = original
        result['equalized'] = equalized
        result['mask'] = mask_copy
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
