# -*- coding: utf-8 -*-

import pickle
import os

import jinja2, math
import matplotlib
from matplotlib import pylab, pyplot, figure, collections, patches, colors, cm
from matplotlib.backends.backend_gtk3agg import FigureCanvasGTK3Agg as FigureCanvas
import numpy as np
import pandas as pd
import scipy
import PIL
from sklearn import svm
from gi.repository import Gtk, Gdk

import xrd
from plots import dual_axes, new_axes
from materials import DummyMaterial

class Cube():
    """Cubic coordinates of a hexagon"""
    @staticmethod
    def from_xy(xy, unit_size):
        x, y = (xy[0], xy[1])
        j = (y/math.sqrt(3)-x)/unit_size
        i = 2*y/math.sqrt(3)/unit_size - j
        i = round(i)
        j = round(j)
        return Cube(i, j, -(i+j))

    def __init__(self, i, j, k, *args, **kwargs):
        self.i = i
        self.j = j
        self.k = k

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


class GtkMapWindow(Gtk.Window):
    """
    A set of plots for interactive data analysis.
    """
    local_mode = False
    map_hexagon = None
    image_hexagon = None
    composite_hexagon = None
    def __init__(self, xrd_map, *args, **kwargs):
        self.xrd_map = xrd_map
        self.currentScan = self.xrd_map.scan(Cube(0, 0, 0))
        return_val = super(GtkMapWindow, self).__init__(*args, **kwargs)
        self.connect('delete-event', Gtk.main_quit)
        # Load icon
        directory = os.path.dirname(os.path.realpath(__file__))
        image = '{0}/images/icon.png'.format(directory)
        self.set_icon_from_file(image)
        self.set_default_size(1000, 1000)
        # Set up the matplotlib features
        fig = figure.Figure(figsize=(13.8, 10))
        self.fig = fig
        fig.figurePatch.set_facecolor('white')
        # Prepare plotting area
        sw = Gtk.ScrolledWindow()
        self.add(sw)
        canvas = FigureCanvas(self.fig)
        canvas.set_size_request(400,400)
        sw.add_with_viewport(canvas)
        self.draw_plots()
        # Connect to keypress event for changing position
        self.connect('key_press_event', self.on_key_press)
        # Connect to mouse click event
        fig.canvas.mpl_connect('button_press_event', self.click_callback)
        return return_val

    def draw_plots(self, scan=None):
        """
        (re)draw the plots on the gtk window
        """
        xrdMap = self.xrd_map
        self.fig.clear()
        # Prepare plots
        self.mapAxes = self.fig.add_subplot(221)
        xrdMap.plot_map(ax=self.mapAxes)
        self.mapAxes.set_aspect(1)
        self.compositeImageAxes = self.fig.add_subplot(223)
        xrdMap.plot_composite_image(ax=self.compositeImageAxes)
        self.scanImageAxes = self.fig.add_subplot(224)
        self.update_plots()

    def update_plots(self):
        """Respond to changes in the selected scan."""
        # Clear old highlights
        if self.map_hexagon:
            self.map_hexagon.remove()
            self.map_hexagon = None
            self.composite_hexagon.remove()
            self.composite_hexagon = None
            self.image_hexagon.remove()
            self.image_hexagon = None
        # Check if a scan should be highlighted
        if self.local_mode:
            activeScan = self.currentScan
        else:
            activeScan = None
        # Plot diffractogram (either bulk or local)
        self.diffractogramAxes = self.fig.add_subplot(222)
        self.diffractogramAxes.cla() # Clear axes
        if activeScan:
            activeScan.plot_diffractogram(ax=self.diffractogramAxes)
        else:
            self.xrd_map.plot_bulk_diffractogram(ax=self.diffractogramAxes)
        # Draw individual scan's image or histogram
        self.scanImageAxes.cla()
        if activeScan:
            activeScan.plot_image(ax=self.scanImageAxes)
        else:
            self.xrd_map.plot_histogram(ax=self.scanImageAxes)
        # Highlight the hexagon on the map and composite image
        if activeScan:
            self.map_hexagon = activeScan.highlight_beam(ax=self.mapAxes)
            self.composite_hexagon = activeScan.highlight_beam(
                ax=self.compositeImageAxes)
            self.image_hexagon = activeScan.highlight_beam(
                ax=self.scanImageAxes)
            self.mapAxes.draw_artist(self.map_hexagon)
        # Force a redraw of the canvas since Gtk won't do it
        self.fig.canvas.draw()

    def on_key_press(self, widget, event, user_data=None):
        oldCoords = self.currentScan.cube_coords
        newCoords = oldCoords
        # Check for arrow keys -> move to new location on map
        if not self.local_mode:
            self.local_mode = True
        elif event.keyval == Gdk.KEY_Left:
            newCoords = oldCoords + Cube(0, 1, -1)
        elif event.keyval == Gdk.KEY_Right:
            newCoords = oldCoords + Cube(0, -1, 1)
        elif event.keyval == Gdk.KEY_Up:
            newCoords = oldCoords + Cube(1, 0, -1)
        elif event.keyval == Gdk.KEY_Down:
            newCoords = oldCoords + Cube(-1, 0, 1)
        elif event.keyval == Gdk.KEY_Escape:
            # Return to bulk view
            self.local_mode = False
        # Check if new coordinates are valid and update scan
        scan = self.xrd_map.scan(newCoords)
        if scan:
            self.currentScan = scan
        self.update_plots()

    def click_callback(self, event):
        """Detect and then update which scan is active."""
        inMapAxes = event.inaxes == self.mapAxes
        inCompositeAxes = event.inaxes == self.compositeImageAxes
        inImageAxes = event.inaxes == self.scanImageAxes
        if (inMapAxes or inCompositeAxes or inImageAxes):
            # Switch to new position on map
            scan = self.xrd_map.scan_by_xy((event.xdata, event.ydata))
            if not self.local_mode:
                self.local_mode = True
            elif scan:
                self.currentScan = scan
        else:
            # Reset local_mode
            self.local_mode = False
        self.update_plots()


class Map():
    """
    A physical sample that gets mapped by XRD, presumed to be circular
    with center and diameter in millimeters. Collimator size given in mm.
    scan_time determines seconds spent at each detector position. Detector
    distance given in cm, frame_size in pixels.
    """
    cmap_name = 'winter'
    THETA1_MIN=0 # Source limits based on geometry
    THETA1_MAX=50
    THETA2_MIN=0 # Detector limits based on geometry
    THETA2_MAX=55
    camera_zoom = 6
    frame_step = 20 # How much to move detector by in degrees
    frame_width = 30 # 2-theta coverage of detector face
    scans = []
    hexagon_patches = None # Replaced by cached versions
    def __init__(self, center=(0, 0), diameter=12.7, collimator=0.5,
                 two_theta_range=None, coverage=1,
                 scan_time=None, sample_name='unknown',
                 detector_distance=20, frame_size=1024,
                 material=DummyMaterial()):
        self.center = center
        self.diameter = diameter
        self.material = material
        self.collimator = collimator
        self.detector_distance=detector_distance
        self.frame_size=frame_size
        if scan_time is not None:
            self.scan_time = scan_time # Seconds at each detector angle
        elif material is not None:
            self.scan_time = material.scan_time
        else:
            # Default value
            self.scan_time = 60
        if two_theta_range is not None:
            self.two_theta_range = two_theta_range
        elif material is not None:
            self.two_theta_range = material.two_theta_range
        else:
            # Default value
            self.two_theta_range = (10, 80)
        self.coverage = coverage
        self.sample_name = sample_name
        self.create_scans()

    @property
    def rows(self):
        """Determine number of rows from collimator size and sample diameter.
        Central spot counts as a row."""
        rows = self.diameter / self.unit_size / math.sqrt(3)
        centerDot = 1
        return math.ceil(rows) + centerDot

    @property
    def unit_size(self):
        """Size of a step in the path."""
        unit_size = math.sqrt(3) * self.collimator / 2
        # Unit size should be bigger if we're not mapping 100%
        unit_size = unit_size / math.sqrt(self.coverage)
        return unit_size

    def create_scans(self):
        """Populate the scans array with new scans in a hexagonal array."""
        self.scans = []
        for idx, coords in enumerate(self.path(self.rows)):
            # Try and determine filename from the sample name
            fileBase = "map-{n:x}".format(n=idx)
            new_scan = MapScan(location=coords, xrd_map=self,
                               material=self.material, filebase=fileBase)
            self.scans.append(new_scan)

    def scan(self, cube):
        """Find a scan in the array give a set of cubic coordinates"""
        result = None
        for scan in self.scans:
            if scan.cube_coords == cube:
                result = scan
                break
        return result

    def scan_by_xy(self, xy):
        """Find the nearest scan by set of xy coords."""
        scan = self.scan(Cube.from_xy(xy, unit_size=self.unit_size))
        return scan

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
        for row in range(1, rows):
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

    def write_slamfile(self, f=None, quiet=False):
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
        # Print summary info
        if not quiet:
            msg = "Running {num} scans. ETA: {time}."
            print(msg.format(num=context['num_scans'], time=context['total_time']))
            frameStart = context['frames'][0]['start']
            frameEnd = context['frames'][-1]['end']
            msg = "Integration range: {start}° to {end}°"
            print(msg.format(start=frameStart, end=frameEnd))
        return f

    def get_context(self):
        """Convert the object to a dictionary for the templating engine."""
        # Estimate the total time
        totalSecs = len(self.scans)*self.scan_time
        days = math.floor(totalSecs/60/60/24)
        remainder = totalSecs - days * 60 * 60 * 24
        hours = math.floor(remainder/60/60)
        remainder = remainder - hours * 60 * 60
        mins = math.floor(remainder/60)
        total_time = "{secs}s ({days}d {hours}h {mins}m)".format(secs=totalSecs,
                                                                 days=days,
                                                                 hours=hours,
                                                                 mins=mins)
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
        # Generate flood and spatial reference files to load
        floodFilename = "{framesize:04d}_{distance:03d}._FL".format(
            distance=self.detector_distance,
            framesize=self.frame_size
        )
        spatialFilename = "{framesize:04d}_{distance:03d}._ix".format(
            distance=self.detector_distance,
            framesize=self.frame_size
        )
        # Prepare context dictionary
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
            'sample_name': self.sample_name,
            'flood_file': floodFilename,
            'spatial_file': spatialFilename,
        }
        for idx, scan in enumerate(self.scans):
            # Prepare scan-specific details
            x, y = scan.xy_coords(unit_size=self.unit_size)
            scan_metadata = {'x': x, 'y': y, 'filename': scan.filebase}
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
        if t2_end - self.THETA1_MAX > self.THETA2_MAX:
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

    def prepare_mapping_data(self):
        """
        Perform initial calculations on mapping data and save results to file.
        """
        self.subtract_backgrounds()
        self.calculate_metrics()
        self.calculate_reliabilities()

    def subtract_backgrounds(self):
        for scan in self.scans:
            scan.subtract_background()

    def calculate_metrics(self):
        """Force recalculation of all metrics in the map."""
        for scan in self.scans:
            scan.cached_data['metric'] = None
            scan.metric

    def calculate_reliabilities(self):
        for scan in self.scans:
            scan.cached_data['reliability'] = None
            scan.reliability

    def save(self, filename=None, overwrite=False):
        """Take cached data and save to disk."""
        # Prepare dictionary of cached data
        data = {
            'diameter': self.diameter,
            'coverage': self.coverage,
            'rows': self.rows,
            'scans': [scan.data_dict for scan in self.scans],
        }
        # Compute filename and Check if file exists
        if filename is None:
            filename = "{sample_name}.map".format(sample_name=self.sample_name)
        if os.path.exists(filename) and not overwrite:
            msg = "Cowardly, refusing to overwrite existing file {}. Pass overwrite=True to force."
            raise IOError(msg.format(filename))
        # Pickle data and write to file
        with open(filename, 'wb') as saveFile:
            pickle.dump(data, saveFile)

    def load(self, filename=None):
        """Load a .map file of previously processed data."""
        # Generate filename if not supplied
        if filename is None:
            filename = "{sample_name}.map".format(sample_name=self.sample_name)
            # Get the data from disk
        with open(filename, 'rb') as loadFile:
            data = pickle.load(loadFile)
        self.diameter = data['diameter']
        self.coverage = data['coverage']
        # Create scan list
        self.scans = []
        # self.create_scans()
        # assert len(self.scans) == len(data['scans'])
        # Restore each scan
        for idx, dataDict in enumerate(data['scans']):
            # scan = self.scans[idx]
            newScan = MapScan(location=dataDict['cube_coords'],
                              xrd_map=self,
                              material=self.material,
                              filebase=dataDict['filebase'])
            newScan.data_dict = dataDict
            self.scans.append(newScan)

    def plot_map_with_image(self, scan=None, alpha=None):
        mapAxes, imageAxes = dual_axes()
        self.plot_map(ax=mapAxes, highlightedScan=scan, alpha=alpha)
        # Plot either the bulk diffractogram or the specific scan requested
        if scan is None:
            self.plot_composite_image(ax=imageAxes)
        else:
            scan.plot_image(ax=imageAxes)
        return (mapAxes, imageAxes)

    def plot_map_with_diffractogram(self, scan=None):
        mapAxes, diffractogramAxes = dual_axes()
        self.plot_map(ax=mapAxes, highlightedScan = scan)
        if scan is None:
            self.plot_bulk_diffractogram(ax=diffractogramAxes)
        else:
            scan.plot_diffractogram(ax=imageAxes)
        return (mapAxes, diffractogramAxes)

    def bulk_diffractogram(self):
        """
        Calculate the bulk diffractogram by averaging each scan weighted
        by reliability.
        """
        bulk_diffractogram = pd.Series()
        scanCount = 0
        # Add a contribution from each map location
        for scan in self.scans:
            if scan.diffractogram_is_loaded:
                scan_diffractogram = scan.diffractogram['counts']
                corrected_diffractogram = scan_diffractogram * scan.reliability
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
        if len(bulk_diffractogram) > 0:
            bulk_diffractogram.plot(ax=ax)
        else:
            print("No bulk diffractogram data to plot")
        self.material.highlight_peaks(ax=ax)
        ax.set_xlabel(r'$2\theta$')
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
        # Plot hexagons
        if not ax:
            # New axes unless one was already created
            ax = new_axes()
        xy_lim = self.xy_lim()
        ax.set_xlim([-xy_lim, xy_lim])
        ax.set_ylim([-xy_lim, xy_lim])
        ax.set_xlabel('mm')
        ax.set_ylabel('mm')
        for scan in self.scans:
            scan.plot_hexagon(ax=ax)
            scan.plot_beam(ax=ax)
        # If a highlighted scan was given, show it in a different color
        if highlightedScan is not None:
            highlightedScan.highlight_beam(ax=ax)
        # Add circle for theoretical edge
        self.draw_edge(ax, color='blue')
        # Add colormap to the side of the axes
        mappable = cm.ScalarMappable(norm=self.material.metric_normalizer, cmap=cmap)
        mappable.set_array(np.arange(0, 2))
        pyplot.colorbar(mappable, ax=ax)
        return ax

    def plot_map_gtk(self):
        """
        Create a gtk window with plots and images for interactive data analysis.
        """
        # Show GTK window
        title = "Maps for sample '{}'".format(self.sample_name)
        window = GtkMapWindow(xrd_map=self, title=title)
        window.show_all()
        Gtk.main()
        # Close the current blank plot
        pyplot.close()

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
                    file_base=scan.filebase
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

    def plot_histogram(self, ax=None):
        metrics = [scan.metric for scan in self.scans]
        min = self.material.metric_normalizer.vmin
        max = self.material.metric_normalizer.vmax
        metrics = np.clip(metrics, min, max)
        weights = [scan.reliability for scan in self.scans]
        if ax is None:
            figure = pyplot.figure()
            ax = pyplot.gca()
        ax.hist(metrics, bins=100, weights=weights)
        ax.set_xlim(min, max)
        ax.set_xlabel('Metric')
        ax.set_ylabel('Occurrences')
        return ax

    def __repr__(self):
        return '<{cls}: {name}>'.format(cls=self.__class__.__name__,
                                        name=self.sample_name)


class MapScan(xrd.XRDScan):
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
    def metric(self):
        """
        Check for a cached metric and if none is found, generate a new
        one.
        """
        metric = self.cached_data.get('metric', None)
        if metric is None:
            if self.diffractogram_is_loaded:
                metric = self.material.mapscan_metric(scan=self)
                self.cached_data['metric'] = metric
            else:
                metric = 0
        return metric

    @metric.setter
    def metric(self, metric):
        self.cached_data['metric'] = metric

    @property
    def reliability(self):
        """Serve up cached value or recalculate if necessary."""
        reliability = self.cached_data.get('reliability', None)
        if reliability is None:
            self.diffractogram
            if self.diffractogram_is_loaded:
                reliability = self.material.mapscan_reliability(scan=self)
                normalizer = self.material.reliability_normalizer
                reliability = normalizer(reliability)
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
        hexagon = patches.RegularPolygon(
            xy=self.xy_coords(),
            numVertices=6,
            radius=radius,
            linewidth=0,
            alpha=self.reliability/3,
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
            color = cmap(self.material.metric_normalizer(metric))
            self.cached_data['color'] = color
        return color

    def axes_title(self):
        """Determine diffractogram axes title from cube coordinates."""
        title = 'XRD Diffractogram at ({i}, {j}, {k}). Metric={metric}'.format(
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
            image = self.image().astype(np.uint16)
            # Add a bias that will be removed after the image is composited
            image = image + 1
        # Calculate padding
        center = self.pixel_coords(height=height, width=width)
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


class DummyMap(Map):
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

    def plot_map(self, *args, **kwargs):
        # Ensure that "diffractogram is loaded" for each scan
        for scan in self.scans:
            scan.diffractogram_is_loaded = True
        return super(DummyMap, self).plot_map(*args, **kwargs)
