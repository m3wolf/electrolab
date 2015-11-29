from collections import namedtuple
import os
import math

import numpy as np
from matplotlib import pyplot

import plots
from hdf import HDFAttribute

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

    def __init__(self, file=None):
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

    @property
    def hdf_node(self):
        return self.image_data

    def plot_image(self, data=None, ax=None, *args, **kwargs):
        """Plot a frame's data image. Use frame.image_data if no data are
        given."""
        if ax is None:
            ax=plots.new_axes()
        if data is None:
            data = self.image_data
        return ax.imshow(self.image_data, *args, cmap='gray', **kwargs)

    def shift_data(self, x_offset, y_offset):
        """Move the image within the view field by the given offsets in pixels.
        New values are filled in with zeroes."""
        # Make sure ints were passed
        original_shape = self.image_data.shape
        # Expand the array to allow for rolling
        new_shapes = (
            (0, abs(y_offset)),
            (0, abs(x_offset))
        )
        new_data = np.pad(self.image_data, new_shapes, mode='constant')
        # Roll along x axis
        new_data = np.roll(new_data, x_offset, axis=1)
        # Roll along y axis
        new_data = np.roll(new_data, y_offset, axis=0)
        # Resize back to original shape
        new_data = np.array(new_data[0:original_shape[0], 0:original_shape[1]])
        self.image_data.write_direct(new_data)

    def create_dataset(self, setname, hdf_group):
        """Save data and metadata to an HDF dataset."""
        attrs = getattr(self.image_data, 'attrs', self._attrs)
        self.image_data = hdf_group.create_dataset(name=setname,
                                                   data=self.image_data)
        # Set metadata attributes
        for attr_name in attrs.keys():
            self.image_data.attrs[attr_name] = attrs[attr_name]

    @classmethod
    def load_from_dataset(Cls, dataset):
        """Accept an HDF5 frame dataset and return a new frame."""
        new_frame = Cls()
        new_frame.image_data = dataset
        return new_frame

    def background_data(self):
        """Return numpy array containing transmission data for reference
        frame."""
        
