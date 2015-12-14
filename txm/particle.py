from collections import namedtuple

import numpy as np

from utilities import xycoord

"""
Describes a single particle detected by image processing (skimage).
"""

BoundingBox = namedtuple('BoundingBox', ('top', 'left', 'bottom', 'right'))

class Particle():
    """A single secondary particle detected by image
    processing. Properties determine by regionprops routine from
    skimage.
    """

    def __init__(self, regionprops, frame):
        """regionprops as output from skimage.measure.regionprops."""
        self.regionprops = regionprops
        self.frame = frame

    def sample_position(self):
        """Convert centroid in pixels to sample position in x, y (µm)."""
        frame_center_pos = self.frame.sample_position
        frame_center_pix = xycoord(self.frame.image_data.shape[1]/2,
                                   self.frame.image_data.shape[0]/2)
        pixel_distance_x = self.centroid().x - frame_center_pix.x
        pixel_distance_y = frame_center_pix.y - self.centroid().y
        um_per_pixel = self.frame.um_per_pixel()
        new_center = xycoord(
            x = frame_center_pos.x + pixel_distance_x * um_per_pixel.x,
            y = frame_center_pos.y + pixel_distance_y * um_per_pixel.y
        )
        return new_center

    def centroid(self):
        center = self.regionprops.centroid
        return xycoord(x=center[1], y=center[0])

    def area(self):
        area = self.regionprops.area
        return area

    def bbox(self):
        return BoundingBox(*self.regionprops.bbox)

    def area(self):
        return self.regionprops.area

    def convex_area(self):
        return self.regionprops.convex_area

    def full_mask(self):
        """Return a mask the same size as frame data with only this particle
        exposed."""
        data = self.frame.image_data
        mask = np.zeros_like(data)
        bbox = self.bbox()
        mask[bbox.top:bbox.bottom, bbox.left:bbox.right] = self.mask()
        return np.logical_not(mask)

    def masked_frame_image(self):
        """Return a masked array for the whole frame with only this particle
        only marked as valid."""
        data = self.frame.image_data
        mask = self.full_mask()
        return np.ma.array(data, mask=mask)

    def mask(self):
        return self.regionprops.image

    def plot_image(self, show_particles=False, *args, **kwargs):
        """Calls the regular plotting routine but with cropped data."""
        return self.frame.plot_image(data=self.crop_image(),
                                     show_particles=show_particles,
                                     *args, **kwargs)
