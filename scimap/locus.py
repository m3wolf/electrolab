# # -*- coding: utf-8 -*-

# import os
# import math

# import numpy
# import scipy
# import pandas
# from matplotlib import patches

# from .coordinates import Cube
# from ..plots import new_axes


# class cached_property():
#     """
#     Calculates a computationally expensive value once and save it for
#     later retrieval. To force recalculation, delete the attribute and it
#     will be calculated again next time it is needed.
#     """
#     def __init__(self, func):
#         self.func = func

#     def is_cached(self):
#         return hasattr(self, 'cached_value')

#     def __get__(self, obj, objtype):
#         # Check for cached value first
#         if not self.is_cached():
#             # If not cached, calculate and store
#             self.cached_value = self.func(obj)
#         # Return value (either cached or calculated)
#         ret = self.cached_value
#         return ret

#     def __set__(self, obj, value):
#         self.cached_value = value

#     def __delete__(self, obj):
#         if self.is_cached():
#             del self.cached_value


# class Locus():
#     """
#     An mapping cell at one X,Y location. Several Locus objects make up a
#     Map object.
#     """
#     IMAGE_HEIGHT = 480  # px
#     IMAGE_WIDTH = 640  # px
#     metric = 0

#     def __init__(self, location, parent_map, filebase):
#         self.cube_coords = location
#         self.parent_map = parent_map
#         self.filebase = filebase

#     @property
#     def signal_level(self):
#         return 1.0

#     @property
#     def reliability(self):
#         """Measure of reliability of the data ranging 0..1"""
#         return 1.0

#     @property
#     def filename(self):
#         filename = "{samplename}-frames/{filebase}.plt".format(
#             samplename=self.parent_map.sample_name,
#             filebase=self.filebase,
#         )
#         return filename

#     @property
#     def data_dict(self):
#         """Return a dictionary of calculated data, suitable for pickling."""
#         dataDict = {
#             'cube_coords': tuple(self.cube_coords),
#             'filebase': self.filebase,
#             'metric': self.metric,
#         }
#         return dataDict

#     def restore_data_dict(self, dataDict):
#         """Restore calulated values from a data dictionary."""
#         self.cube_coords = Cube(*dataDict['cube_coords'])
#         self.filebase = dataDict['filebase']
#         self.metric = dataDict['metric']

#     def xy_coords(self, unit_size=None):
#         """Convert internal coordinates to conventional cartesian coords"""
#         # Get default unit vector magnitude if not given
#         if unit_size is None:
#             unit = self.parent_map.unit_size
#         else:
#             unit = unit_size
#         # Calculate x and y positions
#         cube = self.cube_coords
#         x = unit * 0.5 * (cube.i - cube.j)
#         y = unit * (math.sqrt(3) / 2) * (cube.i + cube.j)
#         return (x, y)

#     def instrument_coords(self, unit_size=1):
#         """
#         Convert internal coordinates to cartesian coordinates relative to
#         the sample stage of the instrument.
#         """
#         xy = self.xy_coords(self.parent_map.unit_size)
#         x = xy[0] + self.parent_map.center[0]
#         y = xy[1] + self.parent_map.center[1]
#         return (x, y)

#     def pixel_coords(self, height, width):
#         """
#         Convert internal coordinates to pixels in an image with given
#         height and width. Assumes the sample center is at the center
#         of the image.
#         """
#         dots_per_mm = self.parent_map.dots_per_mm()
#         xy_coords = self.xy_coords()
#         pixel_coords = {
#             'height': round(height / 2 - xy_coords[1] * dots_per_mm),
#             'width': round(width / 2 + xy_coords[0] * dots_per_mm)
#         }
#         return pixel_coords

#     @property
#     def metric_normalized(self):
#         """Return the metric between 0 and 1."""
#         return self.parent_map.metric_normalizer(self.metric)

#     @property
#     def metric_details(self):
#         """Returns a string describing how the metric was calculated."""
#         return "Not implemented, override in subclasses."

#     def plot_hexagon(self, ax):
#         """Build and plot a hexagon for display on the mapping routine.
#         Return the hexagon patch object."""
#         # Check for cached data
#         radius = 0.595 * self.parent_map.unit_size
#         # Determine how opaque to make the hexagon
#         if self.parent_map.coverage == 1:
#             alpha = self.reliability
#         else:
#             alpha = self.reliability / 3
#         hexagon = patches.RegularPolygon(
#             xy=self.xy_coords(),
#             numVertices=6,
#             radius=radius,
#             linewidth=0,
#             alpha=alpha,
#             color=self.color()
#         )
#         ax.add_patch(hexagon)
#         return hexagon

#     def plot_beam(self, ax):
#         """Build and plot a shape for the actual loaction of the X-ray beam
#         on the mapping routine.
#         Return the patch object."""
#         # Check for cached data
#         diameter = self.parent_map.resolution
#         ellipse = patches.Ellipse(
#             xy=self.xy_coords(),
#             width=diameter,
#             height=diameter,
#             linewidth=0,
#             alpha=self.reliability,
#             color=self.color()
#         )
#         ax.add_patch(ellipse)
#         return ellipse

#     def highlight_beam(self, ax):
#         """Plots a red hexagon to highlight this specific scan."""
#         diameter = self.parent_map.resolution
#         ellipse = patches.Ellipse(
#             xy=self.xy_coords(),
#             width=diameter,
#             height=diameter,
#             linewidth=1,
#             alpha=1,
#             edgecolor='red',
#             facecolor='none'
#         )
#         ax.add_patch(ellipse)
#         return ellipse

#     def color(self):
#         """
#         Use the metric for this material to determine what color this scan
#         should be on the resulting map.
#         """
#         metric = self.metric
#         cmap = self.parent_map.get_cmap()
#         color = cmap(self.parent_map.metric_normalizer(metric))
#         return color

#     def axes_title(self):
#         """Determine axes title from cube coordinates."""
#         title = 'Dataset at ({i}, {j}, {k})'.format(
#             i=self.cube_coords[0],
#             j=self.cube_coords[1],
#             k=self.cube_coords[2],
#         )
#         return title

#     def image(self):
#         """
#         Retrieve the image file taken by the diffractometer.
#         """
#         filename = '{dir}/{file_base}_01.jpg'.format(
#             dir=self.parent_map.directory(),
#             file_base=self.filebase
#         )
#         imageArray = scipy.misc.imread(filename)
#         # Rotate to align with sample coords
#         imageArray = scipy.misc.imrotate(imageArray, 180)
#         return imageArray

#     def plot_image(self, ax=None):
#         """
#         Show the scan's overhead picture on a set of axes.
#         """
#         if ax is None:
#             ax = new_axes()
#         # Calculate axes limit
#         center = self.xy_coords()
#         xMin = center[0] - self.IMAGE_WIDTH / 2 / self.parent_map.dots_per_mm()
#         xMax = center[0] + self.IMAGE_WIDTH / 2 / self.parent_map.dots_per_mm()
#         yMin = center[1] - self.IMAGE_HEIGHT / 2 / self.parent_map.dots_per_mm()
#         yMax = center[1] + self.IMAGE_HEIGHT / 2 / self.parent_map.dots_per_mm()
#         axes_limits = (xMin, xMax, yMin, yMax)
#         try:
#             ax.imshow(self.image(), extent=axes_limits)
#         except FileNotFoundError as file_error:
#             # Plot error message
#             x = (xMax - xMin) / 2
#             y = (yMax - yMin) / 2
#             ax.text(x,
#                     y,
#                     file_error,
#                     horizontalalignment='center',
#                     verticalalignment='center')
#         # Add plot annotations
#         ax.set_title('Micrograph of Mapped Area')
#         ax.set_xlabel('mm')
#         ax.set_ylabel('mm')
#         return ax

#     def padded_image(self, height, width, image=None):
#         """
#         Take the image for this scan and pad the edges to make it new
#         height and width.
#         """
#         if image is None:
#             image = self.image().astype(numpy.uint16)
#             # Add a bias that will be removed after the image is composited
#             image = image + 1
#         # Calculate padding
#         center = self.pixel_coords(height=height, width=width)
#         padLeft = int(center['width'] - self.IMAGE_WIDTH / 2)
#         padRight = int(width - center['width'] - self.IMAGE_WIDTH / 2)
#         padTop = int(center['height'] - self.IMAGE_HEIGHT / 2)
#         padBottom = int(height - center['height'] - self.IMAGE_HEIGHT / 2)
#         # Apply padding
#         paddedImage = numpy.pad(
#             image,
#             ((padTop, padBottom), (padLeft, padRight), (0, 0)),
#             mode='constant'
#         )
#         return paddedImage

#     def padded_image_mask(self, height, width):
#         """
#         Return an array of height x width where any pixels in this scans
#         image are 1 and all other are 0.
#         """
#         scanImage = self.image()
#         scanImage.fill(1)
#         return self.padded_image(height=height,
#                                  width=width,
#                                  image=scanImage)


# class DummyLocus(Locus):
#     """
#     An Locus but with fake data for testing.
#     """

#     @property
#     def signal_level(self):
#         return 1

#     def spline(self, xdata):
#         random = numpy.random.rand(len(xdata))
#         return random

#     @property
#     def diffractogram(self):
#         twoTheta = numpy.linspace(10, 80, 700)
#         counts = numpy.random.rand(len(twoTheta))
#         df = pandas.DataFrame(counts, index=twoTheta, columns=['counts'])
#         return df

#     def image(self):
#         """
#         Retrieve a dummy image file taken by the diffractometer.
#         """
#         directory = os.path.dirname(os.path.realpath(__file__))
#         filename = '{dir}/../images/sample-electrode-image.jpg'
#         filename = filename.format(dir=directory)
#         imageArray = scipy.misc.imread(filename)
#         # Rotate to align with sample coords
#         imageArray = scipy.misc.imrotate(imageArray, 180)
#         return imageArray
