# -*- coding: utf-8 -*-

import os
import math

import numpy
import scipy
import pandas
from matplotlib import patches

from xrd.scan import XRDScan
from mapping.coordinates import Cube
from plots import new_axes

class MapScan(XRDScan):
    """
    An XRD scan at one X,Y location. Several Scan objects make up a
    Sample object.
    """
    IMAGE_HEIGHT = 480 # px
    IMAGE_WIDTH = 640 # px
    def __init__(self, location, xrd_map, filebase, *args, **kwargs):
        self.cube_coords = location
        self.xrd_map = xrd_map
        self.filebase = filebase
        self.filename = "{samplename}-frames/{filebase}.plt".format(
                samplename=self.xrd_map.sample_name,
                filebase=self.filebase,
        )
        return super(MapScan, self).__init__(*args, **kwargs)

    @property
    def data_dict(self):
        """Return a dictionary of calculated data, suitable for pickling."""
        dataDict = {
            'diffractogram': self.diffractogram,
            'cube_coords': tuple(self.cube_coords),
            'filename': self.filename,
            'filebase': self.filebase,
            'metric': self.metric,
            'reliability': self.reliability,
            'spline': self.spline,
        }
        return dataDict

    @data_dict.setter
    def data_dict(self, dataDict):
        """Restore calulated values from a data dictionary."""
        self.diffractogram = dataDict['diffractogram']
        self.diffractogram_is_loaded = dataDict['diffractogram'] is not None
        self.cube_coords = Cube(*dataDict['cube_coords'])
        self.filename = dataDict['filename']
        self.filebase = dataDict['filebase']
        self.metric = dataDict['metric']
        self.reliability = dataDict['reliability']
        self.spline = dataDict['spline']

    def xy_coords(self, unit_size=None):
        """Convert internal coordinates to conventional cartesian coords"""
        # Get default unit vector magnitude if not given
        if unit_size is None:
            unit = self.xrd_map.unit_size
        else:
            unit = unit_size
        # Calculate x and y positions
        cube = self.cube_coords
        x = unit * 1/2 * (cube.i - cube.j)
        y = unit * math.sqrt(3)/2 * (cube.i + cube.j)
        return (x, y)

    def instrument_coords(self, unit_size=1):
        """
        Convert internal coordinates to cartesian coordinates relative to
        the sample stage of the instrument.
        """
        xy = self.xy_coords(self.xrd_map.unit_size)
        x = xy[0] + self.xrd_map.center[0]
        y = xy[1] + self.xrd_map.center[1]
        return (x, y)

    def pixel_coords(self, height, width):
        """
        Convert internal coordinates to pixels in an image with given
        height and width. Assumes the sample center is at the center
        of the image.
        """
        dots_per_mm = self.xrd_map.dots_per_mm()
        xy_coords = self.xy_coords()
        pixel_coords = {
            'height': round(height/2 - xy_coords[1] * dots_per_mm),
            'width': round(width/2 + xy_coords[0] * dots_per_mm)
        }
        return pixel_coords

    @property
    def diffractogram(self):
        """Return a diffractogram based on naming scheme for mapping,
        with caching."""
        # Check for cached version
        if self._df is not None:
            df = self._df
        else:
            # Get file from disk
            filename = self.filename
            df = self.load_diffractogram(filename)
        return df

    @diffractogram.setter
    def diffractogram(self, newDf):
        self._df = newDf

    def load_diffractogram(self, filename):
        # Checking for existance of file allows for partial maps
        if filename is not None and os.path.isfile(filename):
            df = super(MapScan, self).load_diffractogram(filename)
        else:
            df = None
        return df

    @property
    def metric_normalized(self):
        """Return the metric between 0 and 1."""
        return self.xrd_map.metric_normalizer(self.metric)

    @property
    def metric(self):
        """
        Check for a cached metric and if none is found, generate a new
        one.
        """
        metric = self.cached_data.get('metric', None)
        if metric is None:
            if self.diffractogram_is_loaded:
                metric = self.xrd_map.mapscan_metric(scan=self)
                self.cached_data['metric'] = metric
            else:
                metric = 0
        return metric

    @metric.setter
    def metric(self, metric):
        self.cached_data['metric'] = metric

    @property
    def metric_details(self):
        """Returns a string describing how the metric was calculated."""
        return "No additional info"

    @property
    def reliability_raw(self):
        """Acquire un-normalized reliability."""
        reliability = self.xrd_map.mapscan_reliability(scan=self)
        return reliability

    @property
    def reliability(self):
        """Serve up cached value or recalculate if necessary."""
        reliability = self.cached_data.get('reliability', None)
        if reliability is None:
            self.diffractogram
            if self.diffractogram_is_loaded:
                raw = self.reliability_raw
                normalizer = self.xrd_map.reliability_normalizer
                reliability = normalizer(raw)
                self.cached_data['reliability'] = reliability
            else:
                reliability = 0
        return reliability

    @reliability.setter
    def reliability(self, reliability):
        self.cached_data['reliability'] = reliability

    def plot_hexagon(self, ax):
        """Build and plot a hexagon for display on the mapping routine.
        Return the hexagon patch object."""
        # Check for cached data
        radius = 0.595*self.xrd_map.unit_size
        # Determine how opaque to make the hexagon
        if self.xrd_map.coverage == 1:
            alpha = self.reliability
        else:
            alpha = self.reliability/3
        hexagon = patches.RegularPolygon(
            xy=self.xy_coords(),
            numVertices=6,
            radius=radius,
            linewidth=0,
            alpha=alpha,
            color=self.color()
        )
        ax.add_patch(hexagon)
        return hexagon

    def plot_beam(self, ax):
        """Build and plot a shape for the actual loaction of the X-ray beam
        on the mapping routine.
        Return the patch object."""
        # Check for cached data
        diameter = self.xrd_map.collimator
        ellipse = patches.Ellipse(
            xy=self.xy_coords(),
            width=diameter,
            height=diameter,
            linewidth=0,
            alpha=self.reliability,
            color=self.color()
        )
        ax.add_patch(ellipse)
        return ellipse

    def highlight_beam(self, ax):
        """Plots a red hexagon to highlight this specific scan."""
        diameter = self.xrd_map.collimator
        hexagon = patches.Ellipse(
            xy=self.xy_coords(),
            width=diameter,
            height=diameter,
            linewidth=1,
            alpha=1,
            edgecolor='red',
            facecolor='none'
        )
        ax.add_patch(hexagon)
        return hexagon

    def color(self):
        """
        Use the metric for this material to determine what color this scan
        should be on the resulting map.
        """
        color = self.cached_data.get('color', None)
        if color is None:
            # Not cached, so recalculate
            metric = self.metric
            cmap = self.xrd_map.get_cmap()
            color = cmap(self.xrd_map.metric_normalizer(metric))
            self.cached_data['color'] = color
        return color

    def axes_title(self):
        """Determine diffractogram axes title from cube coordinates."""
        title = 'XRD Diffractogram at ({i}, {j}, {k})'.format(
            i=self.cube_coords[0],
            j=self.cube_coords[1],
            k=self.cube_coords[2],
            metric=self.metric,
        )
        return title

    def image(self):
        """
        Retrieve the image file taken by the diffractometer.
        """
        filename = '{dir}/{file_base}_01.jpg'.format(
                dir=self.xrd_map.directory(),
                file_base=self.filebase
            )
        imageArray = scipy.misc.imread(filename)
        # Rotate to align with sample coords
        imageArray = scipy.misc.imrotate(imageArray, 180)
        return imageArray

    def plot_image(self, ax=None):
        """
        Show the scan's overhead picture on a set of axes.
        """
        if ax is None:
            ax = new_axes()
        # Calculate axes limit
        center = self.xy_coords()
        xMin = center[0] - self.IMAGE_WIDTH/2/self.xrd_map.dots_per_mm()
        xMax = center[0] + self.IMAGE_WIDTH/2/self.xrd_map.dots_per_mm()
        yMin = center[1] - self.IMAGE_HEIGHT/2/self.xrd_map.dots_per_mm()
        yMax = center[1] + self.IMAGE_HEIGHT/2/self.xrd_map.dots_per_mm()
        axes_limits = (xMin, xMax, yMin, yMax)
        ax.imshow(self.image(), extent=axes_limits)
        # Add plot annotations
        ax.set_title('Micrograph of Mapped Area')
        ax.set_xlabel('mm')
        ax.set_ylabel('mm')
        return ax

    def padded_image(self, height, width, image=None):
        """
        Take the image for this scan and pad the edges to make it new
        height and width.
        """
        if image is None:
            image = self.image().astype(numpy.uint16)
            # Add a bias that will be removed after the image is composited
            image = image + 1
        # Calculate padding
        center = self.pixel_coords(height=height, width=width)
        padLeft = int(center['width'] - self.IMAGE_WIDTH/2)
        padRight = int(width - center['width'] - self.IMAGE_WIDTH/2)
        padTop = int(center['height'] - self.IMAGE_HEIGHT/2)
        padBottom = int(height - center['height'] - self.IMAGE_HEIGHT/2)
        # Apply padding
        paddedImage = numpy.pad(
            image,
            ((padTop, padBottom), (padLeft, padRight), (0, 0)),
            mode='constant'
        )
        return paddedImage

    def padded_image_mask(self, height, width):
        """
        Return an array of height x width where any pixels in this scans
        image are 1 and all other are 0.
        """
        scanImage = self.image()
        scanImage.fill(1)
        return self.padded_image(height=height,
                                 width=width,
                                 image=scanImage)


class DummyMapScan(MapScan):
    """
    An XRD Scan but with fake data for testing.
    """

    def spline(self, xdata):
        random = numpy.random.rand(len(xdata))
        return random

    @property
    def diffractogram(self):
        twoTheta = numpy.linspace(10, 80, 700)
        counts = numpy.random.rand(len(twoTheta))
        df = pandas.DataFrame(counts, index=twoTheta, columns=['counts'])
        return df

    def image(self):
        """
        Retrieve a dummy image file taken by the diffractometer.
        """
        directory = os.path.dirname(os.path.realpath(__file__))
        filename = '{dir}/../images/sample-electrode-image.jpg'.format(dir=directory)
        imageArray = scipy.misc.imread(filename)
        # Rotate to align with sample coords
        imageArray = scipy.misc.imrotate(imageArray, 180)
        return imageArray
