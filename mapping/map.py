# -*- coding: utf-8 -*-

import math
import os
import pickle

import jinja2

from mapping.coordinates import Cube
from mapping.mapscan import MapScan, DummyMapScan
from materials.material import DummyMaterial

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
    frame_width = 20 # 2-theta coverage of detector face
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
        template = env.get_template('mapping/mapping-template.slm')
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
            msg = "Running {num} scans ({frames} frames each). ETA: {time}."
            print(msg.format(num=context['num_scans'],
                             time=context['total_time'],
                             frames=context['number_of_frames']))
            frameStart = context['frames'][0]['start']
            frameEnd = context['frames'][-1]['end']
            msg = "Integration range: {start}° to {end}°"
            print(msg.format(start=frameStart, end=frameEnd))
        return f

    def get_context(self):
        """Convert the object to a dictionary for the templating engine."""
        # Estimate the total time
        totalSecs = len(self.scans)*self.scan_time*self.get_number_of_frames()
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
            start = self.two_theta_range[0] + frame_num*self.frame_step
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
        # theta2_start = theta2_bottom - self.frame_width/8 + self.frame_width/2
        theta2_start = theta2_bottom + self.frame_width/2
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
        # self.subtract_backgrounds()
        self.calculate_metrics()
        self.calculate_reliabilities()
        self.calculate_colors()

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

    def calculate_colors(self):
        for scan in self.scans:
            scan.cached_data['color'] = None
            scan.color()

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
            scan.plot_diffractogram(ax=diffractogramAxes)
        return (mapAxes, diffractogramAxes)

    def plot_map_with_histogram(self):
        mapAxes, histogramAxes = dual_axes()
        self.plot_map(ax=mapAxes)
        self.plot_histogram(ax=histogramAxes)
        return (mapAxes, histogramAxes)

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
        # If there's space between beam locations, plot beam location
        if self.coverage != 1:
            for scan in self.scans:
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
        window.main()
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
        minimum = self.material.metric_normalizer.vmin
        # minimum = min(metrics)
        maximum = self.material.metric_normalizer.vmax
        # maximum = max(metrics)
        metrics = np.clip(metrics, minimum, maximum)
        weights = [scan.reliability for scan in self.scans]
        if ax is None:
            figure = pyplot.figure()
            ax = pyplot.gca()
        ax.hist(metrics, bins=100, weights=weights)
        ax.set_xlim(minimum, maximum)
        ax.set_xlabel('Metric')
        ax.set_ylabel('Occurrences')
        return ax

    def __repr__(self):
        return '<{cls}: {name}>'.format(cls=self.__class__.__name__,
                                        name=self.sample_name)


class DummyMap(Map):
    """
    Sample that returns a dummy map for testing.
    """
    def bulk_diffractogram(self):
        # Return some random data
        twoTheta = np.linspace(10, 80, num=700)
        counts = np.random.rand(len(twoTheta))
        intensity = pd.DataFrame(counts, index=twoTheta, columns=['counts'])
        return intensity

    def composite_image_with_numpy(self):
        # Stub image to show for layout purposes
        directory = os.path.dirname(os.path.realpath(__file__))
        # Read a cached composite image from disk
        image = scipy.misc.imread('{0}/images/test-composite-image.png'.format(directory))
        return image

    def plot_map(self, *args, **kwargs):
        # Ensure that "diffractogram is loaded" for each scan
        for scan in self.scans:
            scan.diffractogram_is_loaded = True
        return super(DummyMap, self).plot_map(*args, **kwargs)

    def create_scans(self):
        """Populate the scans array with new scans in a hexagonal array."""
        self.scans = []
        for idx, coords in enumerate(self.path(self.rows)):
            # Try and determine filename from the sample name
            fileBase = "map-{n:x}".format(n=idx)
            new_scan = DummyMapScan(location=coords, xrd_map=self,
                                    material=self.material, filebase=fileBase)
            self.scans.append(new_scan)
