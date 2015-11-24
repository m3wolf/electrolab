from collections import namedtuple

import numpy as np

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
    in-memory value.
    """
    def __init__(self, attribute_name, default=0):
        self.attribute_name = attribute_name
        self.attrs[attribute_name] = default

    def __get__(self, obj):
        attrs = getattr(obj.image_data, 'attrs', obj._attrs)
        return attrs[self.attribute_name]

    def __set__(self, obj, value):
        attrs = getattr(obj.image_data, 'attrs', obj._attrs)
        attrs[self.attribute_name] = value


class TXMFrame():

    """A single microscopy image at a certain energy."""
    energy = ImageDatasetAttribute('energy', default=0)
    image_data = None
    sample_position = ImageDatasetAttribute('sample_position',
                                            default=position(0, 0, 0))
    approximate_position = position(0, 0, 0)
    is_background = False
    def __init__(self, file=None):
        if file:
            self.energy = file.energy()
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

    def save_to_dataset(self, dataset):
        """Save data and metadata to an HDF dataset."""
        

    @classmethod
    def load_from_dataset(Cls, dataset):

    """Accept an HDF5 frame dataset and return a new frame."""
        new_frame = Cls()
        new_frame.image_data = dataset.value
        return new_frame
