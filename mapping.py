# -*- coding: utf-8 -*-

import jinja2, math
from matplotlib import pylab, pyplot, figure, collections, patches, colors, cm
from matplotlib.backends.backend_gtk3cairo import FigureCanvasGTK3Cairo as FigureCanvas
import numpy as np
import pandas as pd
import scipy
import PIL
import os
from scipy.stats import linregress
from sklearn import svm
from gi.repository import Gtk, Gdk

def new_axes():
    """Create a new set of matplotlib axes for plotting"""
    height = 5 # in inches
    # Adjust width to accomodate colorbar
    width = height/0.8
    fig = pyplot.figure(figsize=(width, height))
    ax = pyplot.gca()
    return ax

class Cube():
    """Cubic coordinates of a hexagon"""
    def __init__(self, i, j, k, *args, **kwargs):
        self.i = i
        self.j = j
        self.k = k
        # super(Cube, self).__init__(*args, **kwargs)

    def __getitem__(self, key):
        coord_list = [self.i, self.j, self.k]
        return coord_list[key]

    def __add__(self, other):
        new = Cube(
            self.i + other.i,
            self.j + other.j,
            self.k + other.k,
        )
        return new

    def __eq__(self, other):
        result = False
        if self.i == other.i and self.j == other.j and self.k == other.k:
            result = True
        return result

    def __str__(self):
        return "({i}, {j}, {k})".format(i=self.i, j=self.j, k=self.k)

    def __repr__(self):
        return "Cube{0}".format(self.__str__())


class GtkPlots():
    """
    A set of plots for interactive data analysis.
    """
    currentCoords = Cube(0, 0, 0)
    local_mode = False
    def __init__(self, sample, *args, **kwargs):
        self.sample = sample

    def redraw_plots(self, scan=None):
        """
        (re)draw the plots on the gtk window
        """
        sample = self.sample
        # Check if a scan should be highlighted
        if self.local_mode:
            scan = self.sample.scan(self.currentCoords)
        else:
            scan = None

        self.fig.clear()
        # Prepare plots
        mapAxes = self.fig.add_subplot(221)
        sample.plot_map(ax=mapAxes, highlightedScan=scan)
        mapAxes.set_aspect(1)
        compositeImageAxes = self.fig.add_subplot(223)
        sample.plot_composite_image(ax=compositeImageAxes)
        diffractogramAxes = self.fig.add_subplot(222)
        if scan:
            # Draw individual scan's image
            scanImageAxes = self.fig.add_subplot(224)
            scan.plot_image(ax=scanImageAxes)
            scan.plot_diffractogram(ax=diffractogramAxes)
        else:
            sample.plot_bulk_diffractogram(ax=diffractogramAxes)

        # Force a redraw of the canvas since Gtk won't do it
        self.fig.canvas.draw()

    def on_key_press(self, widget, event, user_data=None):
        # Check for arrow keys -> move to new location on map
        if not self.local_mode:
            self.local_mode = True
        elif event.keyval == Gdk.KEY_Left:
            self.currentCoords = self.currentCoords + Cube(-1, 0, 1)
        elif event.keyval == Gdk.KEY_Right:
            self.currentCoords = self.currentCoords + Cube(1, 0, -1)
        elif event.keyval == Gdk.KEY_Up:
            self.currentCoords = self.currentCoords + Cube(0, 1, -1)
        elif event.keyval == Gdk.KEY_Down:
            self.currentCoords = self.currentCoords + Cube(0, -1, 1)
        self.redraw_plots()

    def launch(self):
        self.window = Gtk.Window(
            title="Maps for sample '{}'".format(self.sample.sample_name)
        )
        self.window.connect('delete-event', Gtk.main_quit)
        self.window.set_default_size(1000, 1000)

        # Connect to keypress event for changing position
        self.window.connect('key_press_event', self.on_key_press)

        # Set up the matplotlib features
        self.fig = figure.Figure(figsize=(13.8, 10))

        self.fig.figurePatch.set_facecolor('white')

        sw = Gtk.ScrolledWindow()
        self.window.add(sw)

        canvas = FigureCanvas(self.fig)
        canvas.set_size_request(400,400)
        sw.add_with_viewport(canvas)
        self.redraw_plots()
        self.window.show_all()

        Gtk.main()

class BaseSample():
    """
    A physical sample that gets mapped by XRD, presumed to be circular
    with center and diameter in millimeters. Collimator size given in mm.
    scan_time determines seconds spent at each detector position.
    """
    cmap_name = 'summer'
    two_theta_range = (50, 90) # Detector angle range in degrees
    THETA1_MIN=0 # Source limits based on geometry
    THETA1_MAX=50
    THETA2_MIN=0 # Detector limits based on geometry
    THETA2_MAX=55
    camera_zoom = 6
    frame_step = 20 # How much to move detector by in degrees
    frame_width = 30 # 2-theta coverage of detector face
    scan_time = 300 # 5 minutes per scan
    scans = []
    peak_list = {}
    reliability_peak = None
    hexagon_patches = None # Replaced by cached versions
    # Range to use for normalizing the metric into 0.0 to 0.1
    metric_normalizer = colors.Normalize(0, 1)
    reliability_normalizer = colors.Normalize(0, 1)
    def __init__(self, center=(0, 0), diameter=12.7, collimator=0.5, scan_time=None,
                 rows=None, sample_name='unknown', *args, **kwargs):
        self.center = center
        self.diameter = diameter
        self.collimator = collimator
        if scan_time is not None:
            self.scan_time = scan_time # Seconds at each detector angle
        # Determine number of rows from collimator size
        if rows is None:
            self.rows = math.ceil(diameter/collimator/2)
        else:
            self.rows = rows
        self.sample_name = sample_name
        self.create_scans()
        # return super(BaseSample, self).__init__(*args, **kwargs)

    @property
    def unit_size(self):
        unit_size = self.diameter / self.rows / math.sqrt(3)
        return unit_size

    def create_scans(self):
        """Populate the scans array with new scans in a hexagonal array."""
        self.scans = []
        for idx, coords in enumerate(self.path(self.rows)):
            filename = 'map-{n:x}'.format(
                sample=self.sample_name,
                n=idx
            )
            new_scan = self.XRDScan(coords, filename, sample=self)
            self.scans.append(new_scan)

    def scan(self, cube):
        """Find a scan in the array give a set of cubic coordinates"""
        result = None
        for scan in self.scans:
            if scan.cube_coords == cube:
                result = scan
                break
        return result

    def path(self, rows):
        """Generator gives coordinates for a spiral path around the sample."""
        # Six different directions one can move
        basis_set = {
            'W': Cube(-1, 1, 0),
            'SW': Cube(-1, 0, 1),
            'SE': Cube(0, -1, 1),
            'E': Cube(1, -1, 0),
            'NE': Cube(1, 0, -1),
            'NW': Cube(0, 1, -1)
        }
        # Start in the center
        curr_coords = Cube(0, 0, 0)
        yield curr_coords
        # Spiral through each row
        for row in range(1, rows+1):
            # Move to next row
            curr_coords += basis_set['NE']
            yield curr_coords
            for i in range(0, row-1):
                curr_coords += basis_set['NW']
                yield curr_coords
            # Go around the ring for each basis vector
            for key in ['W', 'SW', 'SE', 'E', 'NE']:
                vector = basis_set[key]
                for i in range(0, row):
                    curr_coords += vector
                    yield curr_coords

    def directory(self):
        return '{samplename}-frames'.format(
            samplename=self.sample_name
        )

    def xy_lim(self):
        return self.diameter/2*1.5

    def write_slamfile(self, f=None):
        """
        Format the sample into a slam file that GADDS can process.
        """
        # Import template
        env = jinja2.Environment(loader=jinja2.PackageLoader('electrolab', ''))
        template = env.get_template('mapping-template.slm')
        self.create_scans()
        context = self.get_context()
        # Create file and directory if necessary
        if f is None:
            directory = self.directory()
            if not os.path.exists(directory):
                os.makedirs(directory)
            filename = '{dir}/{samplename}.slm'.format(
                dir=directory, samplename=self.sample_name
            )
            with open(filename, 'w') as f:
                f.write(template.render(**context))
        else:
            f.write(template.render(**context))
        return f

    def get_context(self):
        """Convert the object to a dictionary for the templating engine."""
        # Estimate the total time
        secs = len(self.scans)*self.scan_time
        total_time = "{0}s ({1:0.1f}h)".format(secs, secs/3600)
        # List of frames to integrate
        frames = []
        for frame_num in range(0, self.get_number_of_frames()):
            start = self.two_theta_range[0] + 2.5 + frame_num*self.frame_step
            end = start + self.frame_step
            frame = {
                'start': start,
                'end': end,
                'number': frame_num,
            }
            frames.append(frame)
        context = {
            'scans': [],
            'num_scans': len(self.scans),
            'frames': frames,
            'frame_step': self.frame_step,
            'number_of_frames': self.get_number_of_frames(),
            'xoffset': self.center[0],
            'yoffset': self.center[1],
            'theta1': self.get_theta1(),
            'theta2': self.get_theta2_start(),
            'aux': self.camera_zoom,
            'scan_time': self.scan_time,
            'total_time': total_time,
            'sample_name': self.sample_name
        }
        for idx, scan in enumerate(self.scans):
            # Prepare scan-specific details
            x, y = scan.xy_coords(unit_size=self.unit_size)
            scan_metadata = {'x': x, 'y': y, 'filename': scan.filename}
            context['scans'].append(scan_metadata)
        return context

    def get_number_of_frames(self):
        theta1 = self.get_theta1()
        num_frames = math.ceil(
            (self.two_theta_range[1]-self.two_theta_range[0])/self.frame_step
        )
        # Check for values outside instrument limits
        t2_start = self.get_theta2_start()
        t2_end = t2_start + num_frames*self.frame_step
        if t2_end > self.THETA2_MAX:
            msg = "2-theta range {given} is outside detector limits: {limits}".format(
                given=self.two_theta_range,
                limits=(self.THETA2_MIN, self.THETA2_MAX))
            raise ValueError(msg)
        return num_frames

    def get_theta2_start(self):
        # Assuming that theta1 starts at highest possible range
        theta1 = self.get_theta1()
        theta2_bottom = self.two_theta_range[0] - theta1
        theta2_start = theta2_bottom - self.frame_width/8 + self.frame_width/2
        return theta2_start

    def get_theta1(self):
        # Check for values outside preset limits
        theta1 = self.two_theta_range[0]
        if theta1 < self.THETA1_MIN:
            msg = "2-theta range {given} is outside source limits: {limits}".format(
                given=self.two_theta_range,
                limits=(self.THETA1_MIN, self.THETA1_MAX))
            raise ValueError(msg)
        elif theta1 > self.THETA1_MAX:
            # Cap the theta1 value at a safety limited maximum
            theta1 = self.THETA1_MAX
        return theta1

    def get_cmap(self):
        """Return a function that converts values in range 0 to 1 to colors."""
        return pyplot.get_cmap(self.cmap_name)

    def plot_map_with_image(self, scan=None, alpha=None):
        fig, (map_axes, diffractogram_axes) = pyplot.subplots(1, 2)
        fig.set_figwidth(13.8)
        fig.set_figheight(5)
        self.plot_map(ax=map_axes, highlightedScan=scan, alpha=alpha)
        # Plot either the bulk diffractogram or the specific scan requested
        if scan is None:
            self.plot_composite_image(ax=diffractogram_axes)
        else:
            scan.plot_diffractogram(ax=diffractogram_axes)
        return fig

    def bulk_diffractogram(self):
        """
        Calculate the bulk diffractogram by averaging each scan weighted
        by reliability.
        """
        bulk_diffractogram = pd.Series()
        scanCount = 0
        # Add a contribution from each map location
        for scan in self.scans:
            try:
                scan_diffractogram = scan.diffractogram()['counts']
            except OSError:
                errorMsg = 'could not load "{filename}" for scan at {coords}'
                errorMsg = errorMsg.format(
                    filename = scan.filename,
                    coords = scan.cube_coords,
                )
                print(errorMsg)
            else:
                corrected_diffractogram = scan_diffractogram * scan.reliability()
                scanCount = scanCount + 1
                bulk_diffractogram = bulk_diffractogram.add(corrected_diffractogram, fill_value=0)
        # Divide by the total number of scans included
        bulk_diffractogram = bulk_diffractogram/scanCount
        return bulk_diffractogram

    def plot_bulk_diffractogram(self, ax=None):
        bulk_diffractogram = self.bulk_diffractogram()
        # Get default axis if none is given
        if ax is None:
            ax = new_axes()
        bulk_diffractogram.plot(ax=ax)
        ax.set_ylabel('counts')
        ax.set_title('Bulk diffractogram')
        return ax

    def plot_map(self, ax=None, highlightedScan=None, alpha=None):
        """
        Generate a two-dimensional map of the electrode surface. Color is
        determined by each scans metric() method. If no axes are given
        via the `ax` argument, a new set will be used. Optionally, a
        highlightedScan can be given which will show up as a different
        color.
        """
        cmap = self.get_cmap()
        # Check for cached hexagons and build if not cached
        if self.hexagon_patches is None:
            x = []
            y = []
            values = []
            colors = []
            alphas = []
            i = 0
            for scan in self.scans:
                i+=1
                coord = scan.xy_coords(self.unit_size)
                x.append(coord[0])
                y.append(coord[1])
                try:
                    color = scan.color()
                except OSError:
                    # Diffractogram cannot be loaded for some reason
                    colors.append('black')
                    alphas.append(1)
                else:
                    colors.append(color)
                    # User can specify an alpha value, otherwise use reliability
                    if alpha is not None:
                        alphas.append(alpha)
                    else:
                        alphas.append(scan.reliability())
            xy = list(zip(x, y))
            # Build and show the hexagons
            self.hexagon_patches = []
            for key, loc in enumerate(xy):
                hexagon = patches.RegularPolygon(xy=loc,
                                                 numVertices=6,
                                                 radius=0.595*self.unit_size,
                                                 linewidth=0,
                                                 alpha=alphas[key],
                                                 color=colors[key])
                self.hexagon_patches.append(hexagon)
        # Plot hexagons
        if not ax:
            # New axes unless one was already created
            ax = new_axes()
        xy_lim = self.xy_lim()
        ax.set_xlim([-xy_lim, xy_lim])
        ax.set_ylim([-xy_lim, xy_lim])
        ax.set_xlabel('mm')
        ax.set_ylabel('mm')
        for hexagon in self.hexagon_patches:
            ax.add_patch(hexagon)
        # If a highlighted scan was given, show it in a different color
        if highlightedScan is not None:
            coords = highlightedScan.xy_coords(self.unit_size)
            hexagon = patches.RegularPolygon(xy=coords,
                                             numVertices=6,
                                             radius=0.595*self.unit_size,
                                             linewidth=0,
                                             alpha=1,
                                             color='red')
            ax.add_patch(hexagon)
        # Add circle for theoretical edge
        self.draw_edge(ax, color='blue')
        # Add colormap to the side of the axes
        mappable = cm.ScalarMappable(norm=self.metric_normalizer, cmap=cmap)
        mappable.set_array(np.arange(0, 2))
        pyplot.colorbar(mappable, ax=ax)
        return ax

    def plot_map_gtk(self):
        """
        Create a gtk window with plots and images for interactive data analysis.
        """
        gtkMap = GtkPlots(sample=self)
        gtkMap.launch()

    def draw_edge(self, ax, color):
        """
        Accept an set of axes and draw a circle for where the theoretical
        edge should be.
        """
        circle = patches.Circle(
            (0, 0),
            radius=self.diameter/2,
            edgecolor=color,
            fill=False,
            linestyle='dashed'
        )
        ax.add_patch(circle)
        return ax

    def dots_per_mm(self):
        """
        Determine the width of the scan images based on sample's camera zoom
        """
        # (dpm taken from camera calibration using quadratic regression)
        regression = lambda x: 3.640*x**2 + 13.869*x + 31.499
        dots_per_mm = regression(self.camera_zoom)
        return dots_per_mm

    def composite_image_with_numpy(self):
        """
        Combine all the individual photos from the diffractometer and
        merge them into one image. Uses numpy to average the pixel values.
        """
        # Check for a cached image to return
        compositeImage = getattr(self, '_numpy_image', None)
         # Check for cached image or create one if not cache found
        if compositeImage is None:
            compositeWidth = int(2 * self.xy_lim() * self.dots_per_mm())
            compositeHeight = compositeWidth
            # Create a new numpy array to hold the composited image
            # (it is unsigned int 16 to not overflow when images are added)
            dtype = np.uint16
            dtypeMax = 65535
            compositeImage = np.ndarray((compositeHeight, compositeWidth, 3),
                                        dtype=dtype)
            # This array keeps track of how many images contribute to each pixel
            counterArray = np.ndarray((compositeHeight, compositeWidth, 3),
                                      dtype=dtype)
            # Set to white by default
            compositeImage.fill(0)
            # Step through each scan
            for scan in self.scans:
                # pad raw image to composite image size
                scanImage = scan.padded_image(height=compositeHeight,
                                              width=compositeWidth)
                # add padded image to composite image
                compositeImage = compositeImage + scanImage
                # create padded image mask
                scanMask = scan.padded_image_mask(height=compositeHeight,
                                                  width=compositeWidth)
                # add padded image mask to counter image
                counterArray = counterArray + scanMask
            # Divide by the total count for each pixel
            compositeImage = compositeImage / counterArray
            # Convert back to a uint8 array for displaying
            compositeImage = compositeImage.astype(np.uint8)
            # Roll over pixels to force white background
            # (bias was added in padded_image method)
            compositeImage = compositeImage - 1
            # Save a cached version
            self._numpy_image = compositeImage
        return compositeImage

    def composite_image_with_pillow(self):
        """
        Combine all the individual photos from the diffractometer and
        merge them into one image. This method uses the pillow library
        instead of numpy arrays.
        """
        # Check for a cached image to return
        compositeImage = getattr(self, '_pillow_image', None)
        if compositeImage is None: # No cached image
            # dpm taken from camera calibration using quadratic regression
            regression = lambda x: 3.640*x**2 + 13.869*x + 31.499
            dots_per_mm = regression(self.camera_zoom)
            size = (
                int(2 * self.xy_lim() * dots_per_mm),
                int(2 * self.xy_lim() * dots_per_mm)
            )
            compositeImage = PIL.Image.new(size=size, mode='RGBA', color='white')
            centerX = size[0]/2
            centerY = size[1]/2
            # Step through each scan
            for scan in self.scans:
                filename = '{dir}/{file_base}_01.jpg'.format(
                    dir=self.directory(),
                    file_base=scan.filename
                )
                rawImage = PIL.Image.open(filename)
                rawImage = rawImage.rotate(180)
                # Create a single frame to average with the current composite
                sampleImage = PIL.Image.new(size=size, mode='RGBA', color=(256, 256, 0, 0))
                pixel_coords = (
                    scan.xy_coords()[0] * dots_per_mm,
                    scan.xy_coords()[1] * dots_per_mm
                )
                center = (
                    size[0]/2 - pixel_coords[0],
                    size[1]/2 + pixel_coords[1]
                )
                box = (
                    int(center[0] - rawImage.size[0]/2),
                    int(center[1] - rawImage.size[1]/2),
                )
                sampleImage.paste(rawImage, box=box)
                # Apply this scan's image to the composite
                compositeImage.paste(rawImage, box=box)
            # Save a cached version
            self._pillow_image = compositeImage
        return compositeImage

    def plot_composite_image(self, ax=None):
        """
        Show the composite micrograph image on a set of axes.
        """
        if ax is None:
            ax = new_axes()
        axis_limits = (
            -self.xy_lim(), self.xy_lim(),
            -self.xy_lim(), self.xy_lim()
        )
        ax.imshow(self.composite_image_with_numpy(), extent=axis_limits)
        # Add plot annotations
        ax.set_title('Micrograph of Mapped Area')
        ax.set_xlabel('mm')
        ax.set_ylabel('mm')
        self.draw_edge(ax, color='red')
        return ax

    def __repr__(self):
        return '<Sample: {name}>'.format(name=self.sample_name)

    class XRDScan():
        """
        An XRD scan at one X,Y location. Several Scan objects make up a
        Sample object.
        """
        IMAGE_HEIGHT = 480 # px
        IMAGE_WIDTH = 640 # px
        peaks_by_hkl = {}
        _df = None # Replaced by load_diffractogram() method
        def __init__(self, location, filename, sample=None, *args, **kwargs):
            self.cube_coords = location
            self.filename = filename
            self.sample = sample
            # return super(BaseSample.XRDScan, self).__init__(*args, **kwargs)

        def xy_coords(self, unit_size=None):
            """Convert internal coordinates to conventional cartesian coords"""
            # Get default unit vector magnitude if not given
            if unit_size is None:
                unit = self.sample.unit_size
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
            xy = self.xy_coords(self.sample.unit_size)
            x = xy[0] + self.sample.center[0]
            y = xy[1] + self.sample.center[1]
            return (x, y)

        def pixel_coords(self, height, width):
            """
            Convert internal coordinates to pixels in an image with given
            height and width. Assumes the sample center is at the center
            of the image.
            """
            dots_per_mm = self.sample.dots_per_mm()
            xy_coords = self.xy_coords()
            pixel_coords = {
                'height': round(height/2 - xy_coords[1] * dots_per_mm),
                'width': round(width/2 + xy_coords[0] * dots_per_mm)
            }
            return pixel_coords

        def diffractogram(self, filename=None):
            """Return a pandas dataframe with the X-ray diffractogram for this
            scan.
            """
            if self._df is None:
                df = self.load_diffractogram(filename)
            else:
                df = self._df
            return df

        def load_diffractogram(self, filename=None):
            if filename is None:
                # Try and determine filename from the sample
                filename = "{samplename}-frames/{filename}.plt".format(
                    samplename=self.sample.sample_name,
                    filename=self.filename
                )
            self._df = pd.read_csv(filename, names=['2theta', 'counts'],
                             sep=' ', comment='!', index_col=0)
            self.subtract_background()
            return self._df

        def subtract_background(self):
            """
            Calculate the baseline for the diffractogram and generate a
            background correction.
            """
            background = self._df.copy()
            # Remove registered peaks
            for key, peak in self.peaks_by_hkl.items():
                background.drop(background[peak[0]:peak[1]].index, inplace=True)
            # Determine a background line from the noise on either side of our peak of interest
            noiseLeft = self._df[42.25:43.75]
            noiseRight = self._df[45.25:46.75]
            linearRegion = pd.concat([noiseLeft, noiseRight])
            regression = linregress(x=linearRegion.index,
                                    y=linearRegion.counts)
            slope = regression[0]
            yIntercept = regression[1]
            # Extrapolate the background for the whole spectrum
            self._df['background'] = self._df.index * slope + yIntercept
            self._df['subtracted'] = self._df.counts-self._df.background
            return self._df

        def metric(self):
            raise NotImplementedError


        def color(self):
            """
            Use the metric for this material to determine what color this scan
            should be on the resulting map.
            """
            cmap = self.sample.get_cmap()
            metric = self.metric()
            color = cmap(self.sample.metric_normalizer(metric))
            return color

        def reliability(self):
            """
            Use peak area to determine how likely this scan is to represent
            sample rather than tape.
            """
            if self.sample.reliability_peak:
                normalize = self.sample.reliability_normalizer
                # Determine peak area for normalization
                df = self.diffractogram()
                peakRange = self.sample.peak_list[self.sample.reliability_peak]
                peak = df.loc[peakRange[0]:peakRange[1], 'subtracted']
                peakArea = np.trapz(y=peak, x=peak.index)
                reliability = normalize(peakArea)
            else:
                reliability = None
            return reliability

        def image(self):
            """
            Retrieve the image file taken by the diffractometer.
            """
            filename = '{dir}/{file_base}_01.jpg'.format(
                    dir=self.sample.directory(),
                    file_base=self.filename
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
            xMin = center[1] - self.IMAGE_WIDTH/2/self.sample.dots_per_mm()
            xMax = center[1] + self.IMAGE_WIDTH/2/self.sample.dots_per_mm()
            yMin = center[0] - self.IMAGE_HEIGHT/2/self.sample.dots_per_mm()
            yMax = center[0] + self.IMAGE_HEIGHT/2/self.sample.dots_per_mm()
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
                image = self.image().astype(np.uint16)
                # Add a bias that will be removed after the image is composited
                image = image + 1
            # Calculate padding
            center = self.pixel_coords(height=height, width=width)
            # print('scan center = {}'.format(self.xy_coords()))
            # print('image center = {}'.format(center))
            padLeft = int(center['width'] - self.IMAGE_WIDTH/2)
            padRight = int(width - center['width'] - self.IMAGE_WIDTH/2)
            padTop = int(center['height'] - self.IMAGE_HEIGHT/2)
            padBottom = int(height - center['height'] - self.IMAGE_HEIGHT/2)
            # Apply padding
            paddedImage = np.pad(
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

        def plot_diffractogram(self, ax=None):

            """
            Plot the XRD diffractogram for this scan. Generates a new set of axes
            unless supplied by the `ax` keyword.
            """
            df = self.diffractogram()
            if not ax:
                fig = pyplot.figure()
                ax = pyplot.gca()
            ax.plot(df.index, df.loc[:, 'counts'])
            ax.set_xlabel(r'$2\theta$')
            ax.set_ylabel('Counts')
            title = 'XRD Diffractogram at ({i}, {j}, {k})'.format(
                i=self.cube_coords[0],
                j=self.cube_coords[1],
                k=self.cube_coords[2]
            )
            ax.set_title(title)
            return ax


class DummySample(BaseSample):
    """
    Sample that returns a dummy map for testing.
    """

    def bulk_diffractogram(self):
        # Return some random data
        twoTheta = np.linspace(10, 80, num=700)
        intensity = pd.DataFrame(twoTheta*50, index=twoTheta)
        return intensity

    def composite_image_with_numpy(self):
        # Stub image to show for layout purposes
        directory = os.path.dirname(os.path.realpath(__file__))
        # Read a cached composite image from disk
        image = scipy.misc.imread('{0}/test-composite-image.png'.format(directory))
        return image

    class XRDScan(BaseSample.XRDScan):
        def metric(self):
            # Just return the distance from bottom left to top right
            p = self.cube_coords[0]
            rows = self.sample.rows
            r = p/2/rows + 0.5
            return r

        def reliability(self):
            return 1


class SolidSolutionSample(BaseSample):
    """
    Class for mapping sample that discharge by the solid solution
    mechanism. Operates by change in 2-theta peak position.
    """
    class XRDScan(BaseSample.XRDScan):
        def metric(self):
            """
            Return the 2θ difference of self.peak1 and self.peak2. Peak
            difference is used to overcome errors caused by shifter
            patterns.
            """
            df = self.diffractogram()
            # Get the 2θ value of peak
            peak2 = self.sample.peak_list[self.sample.metric_peak]
            range2 = df.loc[peak2[0]:peak2[1], 'subtracted']
            theta2 = range2.argmax()
            return theta2


class TwoPhaseSample(BaseSample):
    """
    Class for mapping sample that discharge by a two-phase
    mechanism. Operates by the ratio of peak areas for each phase.
    """
    class XRDScan(BaseSample.XRDScan):
        def metric(self):
            """
            Compare the ratio of two peaks, one for discharged and one for
            charged material.
            """
            df = self.diffractogram()
            # Get peak dataframes for integration
            peakDischarged = df.loc[
                self.sample.peak_list[self.sample.discharged_peak],
                'subtracted'
            ]
            peakCharged = df.loc[
                self.sample.peak_list[self.sample.charged_peak],
                'subtracted'
            ]
            # Integrate peaks
            areaCharged = np.trapz(y=peakCharged, x=peakCharged.index)
            areaDischarged = np.trapz(y=peakDischarged, x=peakDischarged.index)
            # Compare areas of the two peaks
            ratio = areaCharged/(areaCharged+areaDischarged)
            return ratio
