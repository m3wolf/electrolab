from collections import namedtuple
import os

import numpy as np
from matplotlib import pyplot

import plots

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

class ImageDatasetAttribute():
    """A descriptor that returns an HDF5 attribute if possible or else an
    in-memory value. An optional `wrapper` argument will wrap the data
    in this function before getting it.
    """
    def __init__(self, attribute_name, default=0, wrapper=None):
        self.attribute_name = attribute_name
        self.default_value = default
        self.wrapper = wrapper

    def __get__(self, obj, owner):
        attrs = getattr(obj.image_data, 'attrs', obj._attrs)
        value = attrs.get(self.attribute_name, self.default_value)
        if self.wrapper:
            value = self.wrapper(value)
        return value

    def __set__(self, obj, value):
        attrs = getattr(obj.image_data, 'attrs', obj._attrs)
        attrs[self.attribute_name] = value


class TXMFrame():

    """A single microscopy image at a certain energy."""
    image_data = np.zeros(shape=(1024,1024))
    _attrs = {}
    energy = ImageDatasetAttribute('energy', default=0)
    original_filename = ImageDatasetAttribute('original_filename', default=None)
    sample_position = ImageDatasetAttribute('sample_position',
                                            default=position(0, 0, 0),
                                            wrapper=lambda coords: position(*coords))
    approximate_position = ImageDatasetAttribute('approximate_position',
                                                default=position(0, 0, 0),
                                                 wrapper=lambda coords: position(*coords))
    is_background = ImageDatasetAttribute('is_background', default=False)
    def __init__(self, file=None):
        if file:
            self.energy = file.energy()
            self.original_filename = os.path.basename(file.filename)
            self.image_data = file.image_data()
            self.sample_position = file.sample_position()
            self.approximate_position = position(
                round(self.sample_position.x, -1),
                round(self.sample_position.y, -1),
                round(self.sample_position.z, -1)
            )
            self.is_background = file.is_background()

    def __repr__(self):
        name = "<TXMFrame: {energy} eV at ({x}, {y}, {z})"
        return name.format(
            energy=int(self.energy),
            x=self.approximate_position.x,
            y=self.approximate_position.y,
            z=self.approximate_position.z,
        )

    def plot_image(self, ax=None):
        if ax is None:
            ax=plots.new_axes()
        return ax.imshow(self.image_data, cmap='gray')

    def create_dataset(self, setname, hdf_group):
        """Save data and metadata to an HDF dataset."""
        self.image_data = hdf_group.create_dataset(name=setname,
                                                   data=self.image_data)
        # Set metadata attributes
        for attr_name in self._attrs.keys():
            self.image_data.attrs[attr_name] = self._attrs[attr_name]

    @classmethod
    def load_from_dataset(Cls, dataset):
        """Accept an HDF5 frame dataset and return a new frame."""
        new_frame = Cls()
        new_frame.image_data = dataset
        return new_frame
