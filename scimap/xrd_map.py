# -*- coding: utf-8 --*

import warnings

from matplotlib import pyplot, patches, colors, cm
import numpy as np
import scipy
import pandas
from sympy.physics import units

from . import exceptions
from .plots import new_axes, set_outside_ticks
from .xrdstore import XRDStore
from .native_refinement import NativeRefinement
from .utilities import prog, xycoord


class Map():
    """A physical sample that gets mapped by some scientific process,
    presumed to be circular with center and diameter in
    millimeters. Resolution is the size of each cell, given in mm.

    Arguments
    ---------
    - sample_name : A string used for identifying this sample. It is
    used for decided on directory names and guessing the HDF5 file
    name if not explicitely provided.

    - diameter : [deprecated]

    - coverage : [deprecated]

    - hdf_filename : String containing the path to the HDF file. If
      None or omitted, a value will be guessed from the sample_name.

    - resolution : [deprecated]

    """
    cmap_name = 'viridis'
    camera_zoom = 1
    hexagon_patches = None  # Replaced by cached versions
    metric_normalizer = colors.Normalize(0, 1, clip=True)
    metric_name = 'Metric'
    reliability_normalizer = colors.Normalize(0, 1, clip=True)

    def __init__(self, sample_name, diameter=12.7, coverage=1,
                 hdf_filename=None, resolution=1):
        if hdf_filename is None:
            # Guess HDF5 filename from sample_name argument
            self.hdf_filename = sample_name + ".h5"
        else:
            self.hdf_filename = hdf_filename
        self.diameter = diameter
        self.coverage = coverage
        self.sample_name = sample_name
        self.resolution = resolution

    @property
    def name(self):
        return self.sample_name

    @property
    def rows(self):
        """Determine number of rows from resolution and sample diameter.
        Central spot counts as a row."""
        rows = self.diameter / self.unit_size / math.sqrt(3)
        centerDot = 1
        return math.ceil(rows) + centerDot

    @property
    def unit_size(self):
        """Size of a step in the path."""
        unit_size = math.sqrt(3) * self.resolution / 2
        # Unit size should be bigger if we're not mapping 100%
        unit_size = unit_size / math.sqrt(self.coverage)
        return unit_size

    @property
    def loci(self):
        with self.store() as store:
            positions = store.positions
            # step_size = store.step_size
        return positions### * step_size.num

    # def create_loci(self):
    #     """Populate the loci array with new loci in a hexagonal array."""
    #     self.loci = []
    #     for idx, coords in enumerate(self.path(self.rows)):
    #         # Try and determine filename from the sample name
    #         filebase = "map-{n:x}".format(n=idx)
    #         new_locus = self.new_locus(location=coords, filebase=filebase)
    #         self.loci.append(new_locus)

    def locus_by_xy(self, xy):
        """Find the index of the nearest locus by set of xy coords."""
        with self.store() as store:
            pos = store.positions
        distance = np.sqrt(np.sum(np.square(pos-xy), axis=1))
        locus = distance.argmin()
        return locus

    def xy_by_locus(self, locus):
        """Retrieve the x, y position of the locus with the given index."""
        with self.store() as store:
            loc = xycoord(*store.positions[locus,:])
        return loc

    def path(self, *args, **kwargs):
        """Generator gives coordinates for a spiral path around the sample."""
        raise NotImplementedError("Use gadds._path() instead")

    def directory(self):
        raise NotImplementedError("Just don't use it.")

    def get_number_of_frames(self):
        warnings.warn(DeprecationWarning("Use gadds.number_of_frames()"))

    def get_theta2_start(self):
        warnings.warn(DeprecationWarning("Use gadds._detector_start()"))

    def get_theta1(self):
        warnings.warn(DeprecationWarning("Use gadds._source_angle()"))

    def get_cmap(self):
        """Return a function that converts values in range 0 to 1 to colors."""
        # Matplotlib built-in colormaps (viridis et al have been
        # merged in now)
        cmap = pyplot.get_cmap(self.cmap_name)
        return cmap

    def prepare_mapping_data(self):
        """
        Perform initial calculations on mapping data and save results to file.
        """
        self.composite_image()

    def calculate_metrics(self):
        """Force recalculation of all metrics in the map."""
        for scan in prog(self.scans, desc='Calculating metrics'):
            scan.cached_data['metric'] = None
            scan.metric

    def reliabilities(self):
        # Calculate new values
        return [scan.reliability for scan in self.scans]

    def calculate_colors(self):
        for scan in prog(self.scans, desc='Transposing colorspaces'):
            scan.cached_data['color'] = None
            scan.color()

    def subtract_backgrounds(self):
        for scan in prog(self.scans, desc='Fitting background'):
            scan.subtract_background()

    def metric(self, *args, **kwargs):
        """
        Calculate a set of mapping values. Should be implemented by
        subclasses.
        """
        raise NotImplementedError

    def mapscan_metric(self, scan):
        """
        Calculate a mapping value from a MapScan. Should be implemented by
        subclasses.
        """
        raise NotImplementedError

    def save(self, filename=None):
        """Take cached data and save to disk."""
        # Prepare dictionary of cached data
        data = {
            'diameter': self.diameter,
            'coverage': self.coverage,
            'loci': [locus.data_dict for locus in self.loci],
        }
        # Compute filename and Check if file exists
        if filename is None:
            filename = "{sample_name}.map".format(sample_name=self.sample_name)
        # Pickle data and write to file
        with open(filename, 'wb') as saveFile:
            pickle.dump(data, saveFile)

    # def load(self, filename=None):
    #     """Load a .map file of previously processed data."""
    #     # Generate filename if not supplied
    #     if filename is None:
    #         filename = "{sample_name}.map".format(sample_name=self.sample_name)
    #     # Get the data from disk
    #     with open(filename, 'rb') as loadFile:
    #         data = pickle.load(loadFile)
    #     self.diameter = data['diameter']
    #     self.coverage = data['coverage']
    #     # Create scan list
    #     self.create_loci()
    #     # Restore each scan
    #     for idx, dataDict in enumerate(data['loci']):
    #         new_locus = self.loci[idx]
    #         new_locus.restore_data_dict(dataDict)
    #         self.loci.append(new_locus)

    # def fullrange_normalizer(self):
    #     """Determine an appropriate normalizer by looking at the range of
    #     metrics."""
    #     metrics = [locus.metric for locus in self.loci]
    #     new_normalizer = colors.Normalize(min(metrics),
    #                                       max(metrics),
    #                                       clip=True)
    #     return new_normalizer

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
        self.plot_map(ax=mapAxes, highlightedScan=scan)
        if scan is None:
            self.plot_diffractogram(ax=diffractogramAxes)
        else:
            scan.plot_diffractogram(ax=diffractogramAxes)
        return (mapAxes, diffractogramAxes)

    def plot_map_with_histogram(self):
        mapAxes, histogramAxes = dual_axes()
        self.plot_map(ax=mapAxes)
        self.plot_histogram(ax=histogramAxes)
        return (mapAxes, histogramAxes)

    def plot_locus(self, loc, ax, shape, size, color, alpha: float=1):
        """Draw a location on the map.

        Arguments
        ---------
        - loc: tuple of (x, y) values of where the locus should be drawn on `ax`.

        - ax: Matplotlib axes object on which to draw

        - shape: String describing the shape to draw. Choices are "square"/"rect" or "hex".

        - size: How big to make the shape, generally the diameter
          (hex) or length (square or rect).

        - color: Matplotlib color spec for plotting this locus
        """
        loc = xycoord(*loc)
        convertor = colors.ColorConverter()
        color = convertor.to_rgba(color, alpha=alpha)
        if shape in ["square", "rect"]:
            # Adjust xy to account for step size
            corner = xycoord(
                x=loc.x - size / 2,
                y=loc.y - size / 2,
            )
            patch = patches.Rectangle(xy=corner, width=size, height=size,
                                      linewidth=0,
                                      facecolor=color, edgecolor=color)
        elif shape in ['hex']:
            patch = patches.RegularPolygon(xy=loc, numVertices=6,
                                           radius=size/1.60, linewidth=0,
                                           facecolor=color, edgecolor=color)
            pass
        else:
            raise ValueError("Unknown value for shape: '{}'".format(shape))
        # Add patch to the axes
        ax.add_patch(patch)

    def plot_map(self, metric='position', ax=None, phase_idx=0,
                 metric_range=(None, None), highlighted_locus=None,
                 alpha=None, alpha_range=None):
        """Generate a two-dimensional map of the electrode surface. A `metric`
        can and should be given to indicate which quantity should be
        mapped, otherwise the map just shows distance from the origin
        for testing purposes. Color and alpha are determined by the
        Map.metric() method (see its docstring for valid choices).

        Arguments
        ---------
        - ax : A matplotlib Axes object onto which the map will be
          drawn. If omitted, a new Axes object will be created. A new
          colorbar will only be added if this argument is None.

        - phase_idx : Controls which phase will be used for generating
          the metric (eg. cell parameter). Not relevant for all
          metrics.

        - metric : Name of the quantity to be used for determining color.

        - metric_range : Specifies the bounds for mapping. Anything
          outside these bounds will be clipped to the max or min. If
          either value is None, that bound will be set to the range of
          metric values.

        - hightlight_locus : Index of an XRD scan that will receive a
          red circle.

        - alpha : Name of the quantity to be used to determine the
          opacity of each cell. If None, all cells will be opaque.

        - alpha_range : 2-tuple with the values for full transparency
          and full opacity. Anything outside these bounds will be
          clipped.

        """
        cmap = self.get_cmap()
        # Plot loci
        if ax is None:
            # New axes unless one was already created
            add_colorbar = True
            ax = new_axes()
        else:
            add_colorbar = False
        xs, ys = self.loci.swapaxes(0, 1)
        with self.store() as store:
            step_size = float(store.step_size / units.mm)
            layout = store.layout
        # Set axes limits
        ax.set_xlim(min(xs) - step_size, max(xs) + step_size)
        ax.set_ylim(min(ys) - step_size, max(ys) + step_size)
        # ax.set_ylim([-xy_lim, xy_lim])
        ax.set_xlabel('mm')
        ax.set_ylabel('mm')
        metrics = self.metric(phase_idx=phase_idx, param=metric)
        # Normalize the metrics
        metric_normalizer = colors.Normalize(*metric_range)
        # metric_normalizer = normalizer(data=metrics, norm_range=metric_range)
        # Retrieve alpha values
        if alpha is None:
            # Default, all loci are full opaque
            alphas = np.ones_like(metrics)
            alpha_normalizer = colors.Normalize(0, 1, clip=True)
        else:
            alphas = self.metric(phase_idx=phase_idx, param=alpha)
            if alpha_range is None:
                alpha_normalizer = colors.Normalize(min(alphas),max(alphas), clip=True)
            else:
                alpha_normalizer = colors.Normalize(min(alpha_range), max(alpha_range), clip=True)
        # Prepare colors and normalized alpha values
        colors_ = self.get_cmap()(metric_normalizer(metrics))
        alphas = alpha_normalizer(alphas)
        # Plot the actual loci
        for locus, color, alpha_ in zip(self.loci, colors_, alphas):
            self.plot_locus(locus, ax=ax, shape=layout, size=step_size,
                            color=color, alpha=alpha_)
        # If there's space between beam locations, plot beam location
        if self.coverage != 1:
            warnings.warn(UserWarning("coverage not properly displayed"))
        # If a highlighted scan was given, show it in a different color
        if highlighted_locus is not None:
            self.highlight_beam(ax=ax, locus=highlighted_locus)
        # Add circle for theoretical edge
        # self.draw_edge(ax, color='red')
        # Add colorbar to the side of the axes
        if add_colorbar:
            mappable = cm.ScalarMappable(norm=metric_normalizer, cmap=cmap)
            mappable.set_array(np.arange(metric_normalizer.vmin,
                                         metric_normalizer.vmax))
            cb = pyplot.colorbar(mappable, ax=ax)
            if metric in ['a', 'b', 'c']:
                cb.set_label(r'Unit cell parameter {} ($\AA$)'.format(metric))
            elif metric in ['a', 'b', 'c']:
                cb.set_label(r'Unit cell parameter {$^{\circ}$}'.format(metric))
            else:
                cb.set_label(metric)
        # Adjust x and y limits
        xs = self.loci[:,0]
        ys = self.loci[:,1]
        xrange_ = (xs.min() - step_size / 2, xs.max() + step_size / 2)
        yrange_ = (ys.min() - step_size / 2, ys.max() + step_size / 2)
        ax.set_xlim(*xrange_)
        ax.set_ylim(*yrange_)
        ax.set_aspect('equal', adjustable="box")
        # Cosmetic adjustmets to the axes
        set_outside_ticks(ax=ax)
        return ax

    def plot_map_gtk(self, WindowClass=None, *args, **kwargs):
        """Create a gtk window with plots and images for interactive data
        analysis.
        """
        if WindowClass is None:
            from mapping.gtkmapviewer import GtkMapViewer
            WindowClass = GtkMapViewer
        # Show GTK window
        title = "Maps for sample '{}'".format(self.sample_name)
        viewer = WindowClass(parent_map=self, title=title, *args, **kwargs)
        viewer.show()
        # Close the current blank plot
        pyplot.close()

    def draw_edge(self, ax, color):
        """
        Accept an set of axes and draw a circle for where the theoretical
        edge should be.
        """
        circle = patches.Circle(
            (0, 0),
            radius=self.diameter / 2,
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
        return 72 * self.camera_zoom

    def composite_image(self, filename=None, recalculate=False):
        """
        Combine all the individual photos from the diffractometer and
        merge them into one image. Uses numpy to average the pixel values.
        """
        # Check for a cached image to return
        compositeImage = getattr(self, '_numpy_image', None)
        # Check for cached image or create one if not cache found
        if compositeImage is None and not recalculate:
            # Default filename
            if filename is None:
                filename = "{sample_name}-composite.png"
                filename = filename.format(sample_name=self.sample_name)
            if os.path.exists(filename) and not recalculate:
                # Load existing image and cache it
                compositeImage = scipy.misc.imread(filename)
                self._numpy_image = compositeImage
            else:
                # Build composite image
                compositeWidth = int(2 * self.xy_lim() * self.dots_per_mm())
                compositeHeight = compositeWidth
                # Create a new numpy array to hold the composited image
                # (it is unsigned int 16 to not overflow when images are added)
                dtype = np.uint16
                compositeImage = np.ndarray(
                    (compositeHeight, compositeWidth, 3), dtype=dtype
                )
                # Array to keep track of how many images contribute to each px
                counterArray = np.ndarray(
                    (compositeHeight, compositeWidth, 3), dtype=dtype
                )
                # Set to white by default
                compositeImage.fill(0)
                # Step through each scan
                for locus in prog(self.loci, desc="Building Composite Image"):
                    # pad raw image to composite image size
                    locusImage = locus.padded_image(height=compositeHeight,
                                                    width=compositeWidth)
                    # add padded image to composite image
                    compositeImage = compositeImage + locusImage
                    # create padded image mask
                    locusMask = locus.padded_image_mask(height=compositeHeight,
                                                        width=compositeWidth)
                    # add padded image mask to counter image
                    counterArray = counterArray + locusMask
                # Divide by the total count for each pixel
                compositeImage = compositeImage / counterArray
                # Convert back to a uint8 array for displaying
                compositeImage = compositeImage.astype(np.uint8)
                # Roll over pixels to force white background
                # (bias was added in padded_image method)
                compositeImage = compositeImage - 1
                # Save a cached version to memory and disk
                self._numpy_image = compositeImage
                scipy.misc.imsave(filename, compositeImage)
        return compositeImage

    def plot_composite_image(self, ax=None):
        """
        Show the composite micrograph image on a set of axes.
        """
        warnings.warn(UserWarning(), "Not implemented")
        return
        if ax is None:
            ax = new_axes()
        axis_limits = (
            -self.xy_lim(), self.xy_lim(),
            -self.xy_lim(), self.xy_lim()
        )
        ax.imshow(self.composite_image(), extent=axis_limits)
        # Add plot annotations
        ax.set_title('Micrograph of Mapped Area')
        ax.set_xlabel('mm')
        ax.set_ylabel('mm')
        self.draw_edge(ax, color='red')
        return ax

    def plot_histogram(self, metric: str, phase_idx: int=0, ax=None,
                       bins: int=0, weight: str=None,
                       metric_range=(None, None), weight_range=(None, None)):
        """Plot a histogram showing the distribution of the given metric.

        Arguments
        ---------
        - metric : String describing which metric to plot. See
          self.metric() for valid choices.

        - phase_idx : Which phase to use for retrieving the
          metric. Only applicable to things like unit-cell parameters.

        - ax : Matplotlib axes on which to plot.

        - bins : Number of bins in which to distribute the metric
          values. If zero, the number of bins will be determined
          automatically from the number of loci.

        - weight : String describing which metric to use for weighting
          each value. See self.metric() for valid choices. If None,
          all weights will be equal.

        - weight_range : Will be used to normalize the values between
          1 and 0. If not given, then the full range of values will be
          used.
        """
        metrics = self.metric(metric)
        metricnorm = colors.Normalize(*metric_range)
        metricnorm.autoscale_None(metrics)
        np.clip(metrics, metricnorm.vmin, metricnorm.vmax, out=metrics)
        # Guess number of bins'
        if bins is 0:
            bins = int(metrics.shape[0] / 3)
        # Get values for weighting the frequencies
        if weight is not None:
            weights = self.metric(param=weight)
        else:
            weights = np.ones_like(metrics)
            weight_range = (0, 1)
        if weight_range:
            weightnorm = colors.Normalize(*weight_range)
            weights = weightnorm(weights)
        if ax is None:
            ax = new_axes(height=4, width=7)
        # Generate the histogram
        n, bins, patches = ax.hist(metrics, bins=bins, weights=weights)
        # Set the colors based on the metric normalizer
        for patch in patches:
            x_position = patch.get_x()
            cmap = self.get_cmap()
            color = cmap(metricnorm(x_position))
            patch.set_color(color)
        ax.set_xlim(metricnorm.vmin, metricnorm.vmax)
        ax.set_xlabel(metric)
        ax.set_ylabel('Occurrences')
        return ax

    def __repr__(self):
        return '<{cls}: {name}>'.format(cls=self.__class__.__name__,
                                        name=self.sample_name)



class XRDMap(Map):
    """A map using X-ray diffraction to determine cell values. Runs on
    Bruker D8 Discover using GADDS software. Collimator size
    determines resolution in mm. `scan_time` directs how long the
    instrument spends at each point (in seconds).

    The parameter 'phases' is a list of *uninitialized* Phase
    classes. These will be initialized separately for each scan.

    """
    cell_parameter_normalizer = None
    phase_ratio_normalizer = None
    fwhm_normalizer = None
    THETA1_MIN = 0 # Source limits based on geometry
    THETA1_MAX = 50
    THETA2_MIN = 0 # Detector limits based on geometry
    THETA2_MAX = 55
    camera_zoom = 6
    frame_step = 20  # How much to move detector by in degrees
    frame_width = 20  # 2-theta coverage of detector face in degrees
    scan_time = 300  # In seconds
    Phases = []
    background_phases = []

    def __init__(self, *args, collimator=0.5, qrange=None,
                 scan_time=None, detector_distance=20,
                 frame_size=1024, Phases=[], phases=None,
                 background_phases=[], **kwargs):
        # Old-style mapping format deprecation
        if phases is not None:
            warnings.warn(DeprecationWarning(), "Use 'Phases=' instead")
            Phases = phases
        self.collimator = collimator
        self.detector_distance = detector_distance
        self.frame_size = frame_size
        # Checking for non-default lists allows for better subclassing
        if len(Phases) > 0:
            self.Phases = Phases
        if len(background_phases) > 0:
            self.background_phases = background_phases
        if scan_time is not None:
            self.scan_time = scan_time
        if qrange is not None:
            self.qrange = qrange
        # Unless otherwise specified, the collimator sets the resolution
        kwargs['resolution'] = kwargs.get('resolution', collimator)
        # Return parent class init
        return super().__init__(*args, **kwargs)

    def store(self, mode='r'):
        """Get an XRD Store object that saves and retrieves data from the HDF5
        file."""
        return XRDStore(hdf_filename=self.hdf_filename,
                        groupname=self.sample_name, mode=mode)

    def context(self):
        """Convert the object to a dictionary for the templating engine."""
        raise NotImplementedError("Use gadds._context()")

    def write_script(self, file=None, quiet=False):
        """
        Format the sample into a slam file that GADDS can process.
        """
        raise NotImplementedError("Use gadds.write_gadds_script()")

    @property
    def diffractogram(self):
        """Returns self.bulk_diffractogram()."""
        bulk_series = self.bulk_diffractogram()
        df = pandas.DataFrame(bulk_series, columns=['counts'])
        return df

    def get_diffractogram(self, index=None):
        """
        Calculate the bulk diffractogram by averaging each scan weighted
        by reliability.
        """
        with self.store() as store:
            if index is None:
                df = np.average(store.intensities, axis=0)
            else:
                df = store.intensities[index]
        return df
        # bulk_diffractogram = pandas.Series()
        # locusCount = 0
        # Add a contribution from each map location
        # for locus in self.loci:
        #     if locus.diffractogram_is_loaded:
        #         locus_diffractogram = locus.diffractogram['counts']
        #         corrected_diffractogram = locus_diffractogram * locus.reliability
        #         locusCount = locusCount + 1
        #         bulk_diffractogram = bulk_diffractogram.add(corrected_diffractogram, fill_value=0)
        # # Divide by the total number of scans included
        # bulk_diffractogram = bulk_diffractogram / locusCount
        # return bulk_diffractogram

    def plot_diffractogram(self, ax=None, index=None, subtracted=False):
        """Plot a specific diffractogram or an average of all the scans,
        weighted by reliability.

        Arguments
        ---------
        - ax : The matplotlib axes object to plot on to

        - index : Which locus to plot.

        - subtracted : If True, the plot will be shown with background removed.
        """
        # Helper function for determining mean vs single-locus patterns
        def get_values(data, index):
            if index is None:
                return np.average(data, axis=0)
            else:
                return data[index]
        # Retrieve the actual data
        with self.store() as store:
            intensities = get_values(store.intensities, index=index)
            qs = get_values(store.scattering_lengths, index=index)
            bg = get_values(store.backgrounds, index=index)
            fits = get_values(store.fits, index=index)
        # Get default axis if none is given
        if ax is None:
            ax = new_axes()
        # Prepare arrays of data for plotting
        if subtracted:
            observations = intensities - bg
            predictions = fits - bg
        else:
            observations = intensities
            predictions = fits
        residuals = observations - predictions
        # Plot data
        ax.plot(qs, observations, marker="+", linestyle="None")
        ax.plot(qs, predictions)
        ax.plot(qs, bg, color="red")
        ax.plot(qs, residuals, color="cyan")
        # Annotate axes
        ax.set_xlabel(r'Scattering Length (q) $/\AA^{-1}$')
        ax.set_ylabel('Intensity a.u.')
        ax.set_title('Bulk diffractogram')
        # Highlight peaks
        color_list = [
            'green',
            'blue',
            'red',
            'orange'
        ]
        # Helper function to highlight peaks
        def draw_peaks(ax, phase, color):
            """Highlight the expected peak corresponding to this phase."""
            alpha = 0.15
            # Highlight each peak in this phase
            for reflection in phase.reflection_list:
                two_theta = reflection.qrange
                ax.axvspan(two_theta[0], two_theta[1],
                           color=color, alpha=alpha)
        # Highlight phases
        for idx, phase in enumerate(self.Phases):
            draw_peaks(ax=ax, phase=phase, color=color_list[idx])
        # Highlight background
        for phase in self.background_phases:
            draw_peaks(ax=ax, phase=phase, color='grey')
        # Set axes limits
        ax.set_xlim(qs.min(), qs.max())
        return ax

    def highlight_beam(self, ax, locus: int):
        """Draw a border on the map showing which locus is currently selected.

        Arguments
        ---------
        - ax : matplotlib Axes object for plotting. Should already
          have a map on it
        - locus : Index of which locus to highlight. Location will be
          taken from the self.store().
        """
        with self.store() as store:
            diameter = store.step_size.num
            loc = xycoord(*store.positions[locus,:])
        ellipse = patches.Ellipse(
            xy=loc,
            width=diameter,
            height=diameter,
            linewidth=1,
            alpha=1,
            edgecolor='red',
            facecolor='none'
        )
        ax.add_patch(ellipse)
        return ellipse

    def plot_all_diffractograms(self, ax=None, subtracted=False,
                                xstep=0, ystep=5, label_scans=True, *args, **kwargs):
        if ax is None:
            ax = new_axes(width=15, height=15)
        with self.store() as store:
            xs = store.scattering_lengths
            ys = store.intensities
            if subtracted == "background":
                ys = store.backgrounds
            elif subtracted:
                bgs = store.backgrounds
                ys = ys - bgs
            for i, (x, y) in enumerate(zip(xs, ys)):
                x += xstep * i
                y += ystep * i
                ax.plot(x, y, *args, **kwargs)
                if i % 5 == 0 and label_scans:
                    # Plot a text label every 5 plots
                    ax.text(x[-1], y[-1], s=i,
                            verticalalignment="center")
        return ax

    def refine_mapping_data(self, backend='native'):
        """Refine the relevant XRD parameters, such as background, unit-cells,
        etc. This will save the refined data to the HDF file.

        Parameters
        ----------
        backend : str, Refinement

          The style of refinement to perform. The default 'native'
          uses built-in numerical computations. Other methods will be
          added in the future. A user-created subclass of
          ``BaseRefinement`` can also be supplied.
        """
        # Empty arrays to hold results
        bgs = []
        fits = []
        all_cells = []
        failed = []
        goodness = []
        fractions = []
        # Retrieve the refinement method based on the given parameter
        backends = {
            'native': NativeRefinement
        }
        if backend in backends.keys():
            Refinement = backends[backend]
        elif issubclass(backend, BaseRefinement):
            Refinement = backend
        else:
            # Invalid refinement backend, throw exception
            msg = "Invalid backend {given}. Must be one of {strings} "
            msg += "or subclass of ``BaseRefinement``."
            msg = msg.format(given=backend, string=backends.keys())
            raise ValueError(msg)
        # Open the data storage and start the refining
        with self.store(mode='r+') as store:
            scans = zip(store.scattering_lengths[:], store.intensities[:])
            total = len(store.scattering_lengths)
            for idx, (qs, Is) in enumerate(prog(scans, total=total, desc="Refining")):
                phases = [P() for P in self.Phases]
                refinement = Refinement(phases=phases)
                # Refine background
                bg = refinement.refine_background(scattering_lengths=qs,
                                                  intensities=Is,
                                                  k=3, s=len(qs)/20)
                bgs.append(bg)
                # Refine unit-cell parameters
                subtracted = Is - bg
                try:
                    residuals = refinement.refine_unit_cells(
                        scattering_lengths=qs,
                        intensities=subtracted,
                        quiet=True
                    )
                except (exceptions.RefinementError, exceptions.UnitCellError, ZeroDivisionError):
                    failed.append(idx)
                    goodness.append(np.nan)
                else:
                    goodness.append(residuals)
                finally:
                    phases = tuple(p.unit_cell.as_tuple() for p in refinement.phases)
                    all_cells.append(phases)
                # Refine the phase fractions
                frac = refinement.refine_phase_fractions(
                    scattering_lengths=qs,
                    intensities=subtracted
                )
                fractions.append(frac)
                # Append the fitted diffraction pattern
                fits.append(np.zeros_like(qs))
                # fits.append(refinement.predict(qs))
            # Store refined data for later
            store.backgrounds = np.array(bgs)
            store.cell_parameters = np.array(all_cells)
            store.fits = np.array(fits)
            store.phase_fractions = fractions
            # Normalize goodnesses of fit values to be between 0 and 1
            max_ = np.nanmax(goodness)
            min_ = np.nanmin(goodness)
            goodness = (max_ - goodness) / (max_ - min_)
            store.goodness = np.nan_to_num(goodness)
        # Alert the user of failed refinements
        if failed:
            msg = "Could not refine unit cell for loci: {}".format(failed)
            warnings.warn(msg, RuntimeWarning)

    # def set_metric_phase_ratio(self, phase_idx=0):
    #     """Set the plotting metric as the proportion of given phase."""
    #     for locus in prog(self.loci, desc='Calculating metrics'):
    #         phase_scale = locus.phases[phase_idx].scale_factor
    #         total_scale = sum([phase.scale_factor for phase in locus.phases])
    #         locus.metric = phase_scale / total_scale

    def valid_metrics(self):
        """Return a list of the available metrics that a user can map. See
        XRDMap.metric() docstring for a full explanation.
        """
        valid = ['a', 'b', 'c', 'alpha', 'beta', 'gamma',
                 'integral', 'goodness', 'position', 'None']
        return valid

    def metric(self, param, phase_idx=0, locus=None):
        """Calculate a mapping value as the parameter (eg. unit-cell a) for
        given phase index `phaseidx`. Valid parameters:
        - Unit cell parameters: 'a', 'b', 'c', 'alpha', 'beta', 'gamma'
        - 'integral' to indicate total integrated signal after bg subtraction
        - 'goodness' to use the quality of fit determined during refinement
        - 'position' to give the distance from the origin (for testing purposes)
        - 'phase_ratio' to give the fraction of the given ``phase_idx``
        - 'None' or None to give an array of 1's

        Returns
        -------
        A numpy array with the requested metrics. If `locus` is None,
        this array will contain all loci, otherwise it will be only
        the requested locus.
        """
        # Check for unit-cell parameters
        UNIT_CELL_PARAMS = ['a', 'b', 'c', 'alpha', 'beta', 'gamma']
        if param in UNIT_CELL_PARAMS:
            param_idx = UNIT_CELL_PARAMS.index(param)
            with self.store() as store:
                metric = store.cell_parameters[:,phase_idx,param_idx]
        elif param == 'integral':
            with self.store() as store:
                q = store.scattering_lengths
                I = store.subtracted
                metric = scipy.integrate.trapz(y=I, x=q, axis=1)
        elif param == "position":
            with self.store() as store:
                locs = store.positions
            metric = np.sqrt(locs[:,0]**2 + locs[:,1]**2)
        elif param in ['goodness']:
            # Just return the requested store attribute
            with self.store() as store:
                metric = getattr(store, param)
        elif param == 'phase_fraction':
            with self.store() as store:
                metric = store.phase_fractions[:,phase_idx]
        elif param in ['None', None]:
            # Return a dummy array with all 1's
            with self.store() as store:
                metric = np.ones(shape=store.scattering_lengths.shape[0])
        else:
            raise ValueError("Unknown param: {}".format(param))
        # Filter by requested locus
        if locus is not None:
            metric = metric[locus:locus+1]
        return metric

    def plot_phase_ratio(self, phase_idx=0, *args, **kwargs):
        warnings.warn(UserWarning("Use `Map.plot_map(metric='phase_ratio')` instead"))
        # """Plot a map of the ratio of the given phase index to all the phases"""
        # self.set_metric_phase_ratio(phase_idx=0)
        # self.metric_name = 'Phase ratio'
        # # Determine normalization range
        # if self.phase_ratio_normalizer is None:
        #     self.metric_normalizer = self.fullrange_normalizer()
        #     self.calculate_normalizer()
        # else:
        #     self.metric_normalizer = self.phase_ratio_normalizer
        # # Plot the map
        # ax = self.plot_map(*args, **kwargs)
        # return ax

    # def set_metric_cell_parameter(self, parameter='a', phase_idx=0):
    #     for locus in prog(self.loci, desc='Calculating cell parameters'):
    #         phase = locus.phases[phase_idx]
    #         locus.metric = getattr(phase.unit_cell, parameter)

    def plot_cell_parameter(self, parameter='a', phase_idx=0, *args, **kwargs):
        warnings.warn(UserWarning(
            "Use `Map.plot_map(metric='{}')` instead".format(parameter)
        ))
        # self.set_metric_cell_parameter(parameter, phase_idx)
        # self.metric_name = 'Unit-cell parameter {0} Å'.format(parameter)
        # # Determine normalization range
        # if self.cell_parameter_normalizer is None:
        #     self.metric_normalizer = self.fullrange_normalizer()
        # else:
        #     self.metric_normalizer = self.cell_parameter_normalizer
        # # Now plot the map
        # return self.plot_map(*args, **kwargs)

    # def set_metric_fwhm(self, phase_idx=0):
    #     for locus in prog(self.loci, desc='Calculating peak widths'):
    #         locus.metric = locus.refinement.fwhm()

    def plot_fwhm(self, phase_idx=0, *args, **kwargs):
        warnings.warn(UserWarning("Use `Map.plot_map(metric='fwhm')` instead"))
        # self.set_metric_fwhm(phase_idx=phase_idx)
        # self.metric_name = 'Full-width half max (°)'
        # # Determine normalization range
        # if self.fwhm_normalizer is None:
        #     self.metric_normalizer = self.fullrange_normalizer()
        # else:
        #     self.metric_normalizer = self.fwhm_normalizer
        # # Now plot the map
        # return self.plot_map(*args, **kwargs)

    def plot_map_gtk(self, *args, **kwargs):
        from .gtkmapviewer import GtkXrdMapViewer
        return super().plot_map_gtk(WindowClass=GtkXrdMapViewer, *args, **kwargs)

    def dots_per_mm(self):
        """Determine the width of the scan images based on sample's camera
        zoom (dpm taken from camera calibration using quadratic
        regression on Bruker D8 Discover Series II)
        """
        def regression(x):
            return 3.640 * x**2 + 13.869 * x + 31.499
        dots_per_mm = regression(self.camera_zoom)
        return dots_per_mm

    def prepare_mapping_data(self):
        warnings.warn(UserWarning("Use `Map.refine_mapping_data() instead`"))
        # for locus in self.loci:
        #     locus.load_diffractogram()
        # self.refine_scans()
        # return super().prepare_mapping_data()

    def refine_scans(self):
        """
        Refine a series of parameters on each scan. Continue if an
        exceptions.RefinementError occurs.
        """
        warnings.warn(UserWarning("Use `Map.refine_mapping_data() instead`"))
        # for locus in prog(self.loci, desc='Reticulating splines'):
        #     try:
        #         current_step = 'background'
        #         locus.refinement.refine_background()
        #         current_step = 'displacement'
        #         locus.refinement.refine_displacement()
        #         current_step = 'peak_widths'
        #         locus.refinement.refine_peak_widths()
        #         current_step = 'unit cells'
        #         locus.refinement.refine_unit_cells()
        #         # if len(locus.phases) > 2:
        #         current_step = 'scale factors'
        #         locus.refinement.refine_scale_factors()
        #     except exceptions.SingularMatrixError as e:
        #         # Display an error message on exception and then coninue fitting
        #         msg = "{coords}: {msg}".format(coords=locus.cube_coords, msg=e)
        #         print(msg)
        #     except exceptions.DivergenceError as e:
        #         msg = "{coords}: DivergenceError while refining {step}".format(
        #             coords=locus.cube_coords,
        #             step=current_step
        #         )
        #         print(msg)
        #     except exceptions.PCRFileError as e:
        #         msg = "Could not read resulting pcr file: {}".format(e)
        #         print(msg)

class PeakPositionMap(Map):
    """A map based on the two-theta position of the diagnostic reflection
    in the first phase.
    """

    def mapscan_metric(self, scan):
        """
        Return the 2θ difference of self.peak1 and self.peak2. Peak
        difference is used to overcome errors caused by shifter
        patterns.
        """
        main_phase = scan.phases[0]
        two_theta_range = main_phase.diagnostic_reflection.two_theta_range
        metric = scan.peak_position(two_theta_range)
        return metric


class PhaseRatioMap(Map):

    def mapscan_metric(self, scan):
        """Compare the ratio of two peaks, one for discharged and one for
        charged material.
        """
        # Query refinement for the contributions from each phase
        contributions = [phase.scale_factor for phase in scan.phases]
        total = sum(contributions)
        if total > 0:  # Avoid div by zero
            ratio = contributions[0] / sum(contributions)
        else:
            ratio = 0
        return ratio

    def mapscan_reliability(self, scan):
        """Determine the maximum total intensity of signal peaks."""
        scale_factors = [phase.scale_factor for phase in scan.phases]
        # area1 = self._phase_signal(scan=scan, phase=scan.phases[0])
        # area2 = self._phase_signal(scan=scan, phase=scan.phases[1])
        return sum(scale_factors)

    def _phase_signal(self, scan, phase):
        peak = phase.diagnostic_reflection.two_theta_range
        area = scan.peak_area(peak)
        return area

    def _peak_position(self, scan, phase):
        peak = phase.diagnostic_reflection.two_theta_range
        angle = scan.peak_position(peak)
        return angle

    def metric_details(self, scan):
        """
       Return a string with the measured areas of the two peaks.
       """
        area1 = self._phase_signal(scan=scan, phase=self.phase_list[0])
        angle1 = self._peak_position(scan=scan, phase=self.phase_list[0])
        area2 = self._phase_signal(scan=scan, phase=self.phase_list[1])
        angle2 = self._peak_position(scan=scan, phase=self.phase_list[1])
        template  = "Area 1 ({angle1:.02f}°): {area1:.03f}\n"
        template += "Area 2 ({angle2:.02f}°): {area2:.03f}\n"
        template += "Sum: {total:.03f}"
        msg = template.format(area1=area1, angle1=angle1,
                              area2=area2, angle2=angle2,
                              total=area1 + area2)
        return msg


class FwhmMap(Map):
    def mapscan_metric(self, scan):
        """
        Return the full-width half-max of the diagnostic peak in the first
        phase.
        """
        angle = sum(scan.phases[0].diagnostic_reflection.two_theta_range) / 2
        fwhm = scan.refinement.fwhm(angle)
        return fwhm
