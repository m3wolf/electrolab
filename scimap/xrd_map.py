# -*- coding: utf-8 --*

import warnings
import logging
log = logging.getLogger(__name__)
import re

from matplotlib import pyplot, patches, colors, cm, rcParams
import numpy as np
import scipy
import pandas as pd

from . import exceptions
from .units_ import units
from .plots import new_axes, set_outside_ticks, dual_axes
from .xrdstore import XRDStore
from .base_refinement import BaseRefinement
from .fullprof_refinement import FullprofRefinement
from .native_refinement import NativeRefinement
from .pawley_refinement import PawleyRefinement
from .utilities import prog, xycoord, q_to_twotheta, twotheta_to_q


class Map():
    """A physical sample that gets mapped by some scientific process,
    presumed to be circular with center and diameter in
    millimeters. Resolution is the size of each cell, given in mm.
    
    Arguments
    ---------
    sample_name : str
      A string used for identifying this sample. It is used for
      decided on directory names and guessing the HDF5 file name if not
      explicitely provided.
    diameter : optional
      [deprecated]
    coverage : optional
      [deprecated]
    hdf_filename : str
      String containing the path to the HDF file. If None or omitted,
      a value will be guessed from the sample_name.
    resolution : 
      [deprecated]
    
    """
    cmap_name = 'viridis'
    camera_zoom = 1
    hexagon_patches = None  # Replaced by cached versions
    metric_normalizer = colors.Normalize(0, 1, clip=True)
    metric_name = 'Metric'
    reliability_normalizer = colors.Normalize(0, 1, clip=True)
    
    def __init__(self, sample_name, diameter=12.7, coverage=1,
                 hdf_filename=None, tube='Cu', resolution=1):
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
            positions = store.positions.value
            # step_size = store.step_size
        return positions### * step_size.num
    
    def locus_by_xy(self, xy):
        """Find the index of the nearest locus by set of xy coords."""
        xy = np.array(xy)
        with self.store() as store:
            pos = store.positions.value
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
    
    def get_cmap(self, cmap=None, metric=None):
        """Return a function that converts values in range 0 to 1 to colors.
        
        If a cmap is not explicitely given, a default will be used
        based on the metric given based on the dictionary
        ``self.default_cmaps``.
        
        Parameters
        ----------
        cmap : str, optional
          The name of the colormap to be passed to pyplot.get_cmap. If
          omitted, a default will be chosen based on ``metric``.
        metric : str, optional
          A metric name to be used for looking up a default colormap.
        
        """
        match = self.metric_re.match(metric)
        if match:
            metric = match.group('param')
        default_cmaps = {
            'a': 'viridis',
            'b': 'viridis',
            'c': 'viridis',
            'alpha': 'viridis',
            'beta': 'viridis',
            'gamma': 'viridis',
            'phase_fraction': 'plasma',
            'broadenings': 'magma',
        }
        # Matplotlib built-in colormaps (viridis et al have been
        # merged in now)
        if cmap is None:
            _cmap = default_cmaps.get(metric, 'viridis')
        else:
            _cmap = cmap
        _cmap = pyplot.get_cmap(_cmap)
        return _cmap
    
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
    
    def subtract_backgrounds(self, bkg, loc_idx=None):
        '''Subtracts the background from patterns.
        
        Background subtracted intensities will be written to already
        existing hdf5 file as intensities_subtracted.
        
        Parameters
        ----------
        bkg : np.ndarray
          Refers to the 1d diffraction pattern that is to be
          subtracted from the intensity data.
        
        loc_idx : np.ndarray, optional
          Refers to the 1d diffraction pattern that will have its background
          subtracted. If omitted all patterns from the map will be
          background subtracted.
        
        '''
        data_in = self.store()
        Is = data_in.intensities
        
        if loc_idx is not None:
            Is = Is[loc_idx]
       
        bkg_sub = Is - bkg
        data_in.close()
        with self.store(mode='r+') as store:
            store.intensities_subtracted = bkg_sub
            
    
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
    
    def plot_map_with_histogram(self, metric='position', axs=None,
                                metric_range=None, highlighted_locus=None,
                                alpha=None, alpha_range=None, cmap=None):
        """Generate a 2D map and a histogram of the electrode surface.
        
        A `metric` can and should be given to indicate which quantity
        should be mapped, otherwise the map just shows distance from
        the origin for testing purposes. Color and alpha are
        determined by the Map.metric() method (see its docstring for
        valid choices).
        
        Arguments
        ---------
        axs : 2-tuple, optional
          A 2-tuple of matplotlib Axes object onto which the map and
          histogram will be drawn. If omitted, a new Axes object will
          be created. A new colorbar will only be added if this
          argument is None.
        metric : str, optional
          Name of the quantity to be used for determining color.
        metric_range : 2-tuple, optional
          Specifies the bounds for mapping. Anything outside these
          bounds will be clipped to the max or min. If omitted, the
          2nd and 98th percentile will be used.
        hightlight_locus : int, optional
          Index of an XRD scan that will receive a red circle.
        alpha : str, optional
          Name of the quantity to be used to determine the opacity of
          each cell. If None, all cells will be opaque.
        alpha_range : 2-tuple, optional
          2-tuple with the values for full transparency and full
          opacity. Anything outside these bounds will be clipped.
        
        """
        # Prepare the axes
        if axs is None:
            mapAxes, histogramAxes = dual_axes()
        else:
            mapAxes, histogramAxes = axs
        # Prepare plotting arguments
        # Do the plotting
        self.plot_map(ax=mapAxes, metric=metric,
                      metric_range=metric_range, alpha=alpha,
                      alpha_range=alpha_range, cmap=cmap)
        self.plot_histogram(ax=histogramAxes,
                            metric=metric, metric_range=metric_range,
                            weight=alpha, weight_range=alpha_range,
                            cmap=cmap)
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
        else:
            raise ValueError("Unknown value for shape: '{}'".format(shape))
        # Add patch to the axes
        ax.add_patch(patch)
    
    def plot_map(self, metric='position', ax=None, phase_idx=0,
                 metric_range=None, highlighted_locus=None,
                 alpha=None, alpha_range=None, cmap=None):
        """Generate a two-dimensional map of the electrode surface.
        
        A `metric` can and should be given to indicate which quantity
        should be mapped, otherwise the map just shows distance from
        the origin for testing purposes. Color and alpha are
        determined by the Map.metric() method (see its docstring for
        valid choices).
        
        Arguments
        ---------
        ax : optional
          A matplotlib Axes object onto which the map will be
          drawn. If omitted, a new Axes object will be created. A new
          colorbar will only be added if this argument is None.
        phase_idx : int, optional
          Controls which phase will be used for generating the metric
          (eg. cell parameter). Not relevant for all metrics.
        metric : str, optional
          Name of the quantity to be used for determining color.
        metric_range : 2-tuple, optional
          Specifies the bounds for mapping. Anything outside these
          bounds will be clipped to the max or min. If omitted, the
          2nd and 98th percentile will be used.
        hightlight_locus : int, optional
          Index of an XRD scan that will receive a red circle.
        alpha : str, optional
          Name of the quantity to be used to determine the opacity of
          each cell. If None, all cells will be opaque.
        alpha_range : 2-tuple, optional
          2-tuple with the values for full transparency and full
          opacity. Anything outside these bounds will be clipped.
        cmap : str, optional
          Matplotlib colormap string for converting numeric values
          into colors.
        
        """
        cmap_ = self.get_cmap(cmap, metric=metric)
        # Plot loci
        add_colorbar = False
        if ax is None:
            # New axes unless one was already created
            ax = new_axes()
            add_colorbar = True
        xs, ys = np.swapaxes(self.loci, 0, 1)
        with self.store() as store:
            step_size = float(store.step_size / store.position_unit)
            layout = store.layout
        # Set axes limits
        ax.set_xlim(min(xs) - step_size, max(xs) + step_size)
        ax.set_ylim(min(ys) - step_size, max(ys) + step_size)
        # ax.set_ylim([-xy_lim, xy_lim])
        ax.set_xlabel('mm')
        ax.set_ylabel('mm')
        metrics = self.metric(param=metric)
        # Set default mapping range if not given
        if metric_range is None:
            metric_range = (np.percentile(metrics, 2),
                            np.percentile(metrics, 98))
        # Normalize the metrics
        metric_normalizer = colors.Normalize(*metric_range)
        # metric_normalizer = normalizer(data=metrics, norm_range=metric_range)
        # Retrieve alpha values
        if alpha is None:
            # Default, all loci are full opaque
            alphas = np.ones_like(metrics)
            alpha_normalizer = colors.Normalize(0, 1, clip=True)
        else:
            alphas = self.metric(param=alpha)
            if alpha_range is None:
                alpha_normalizer = colors.Normalize(min(alphas),max(alphas), clip=True)
            else:
                alpha_normalizer = colors.Normalize(min(alpha_range), max(alpha_range), clip=True)
        # Prepare colors and normalized alpha values
        colors_ = cmap_(metric_normalizer(metrics))
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
            if type(highlighted_locus)==int:
                self.highlight_beam(ax=ax, locus=highlighted_locus)
            elif type(highlighted_locus)==list:
                for locus in highlighted_locus:
                    self.highlight_beam(ax=ax, locus=locus)
            else:
                raise TypeError("highlighted_locus must be of type int or list")
                    
        # Add circle for theoretical edge
        # self.draw_edge(ax, color='red')
        # Add colorbar to the side of the axes
        if add_colorbar:
            mappable = cm.ScalarMappable(norm=metric_normalizer, cmap=cmap_)
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
    
    def plot_distance_image(self, center=(0, 0), ax=None, *args, **kwargs):
        """Plot an image with spectra sorted by distance from center.
        
        Parameters
        ----------
        center : 2-tuple, optional
          (x, y) coordinates used to calculate the center of the
          image. Units should match those used for plotting maps.
        ax : matplotlib.Axes
          Axes to receive the plot.
        *args, **kwargs :
          Passed on to the ``ax.imshow()`` method.
        
        Returns
        -------
        im : 
          Matplotlib artist that created the image.
        """
        if ax is None:
            ax = pyplot.gca()
        # Prepare list of distances for each locus
        loci = np.array(self.loci)
        r = np.sqrt((loci[:,0]-center[0])**2 + (loci[:,1]-center[1])**2)
        r = np.round(r, decimals=3)
        df = pd.DataFrame(data=r)
        df.groupby(0)
        # Find the mean pattern for each distance
        all_Is = []
        all_ds = []
        with self.store() as store:
            for d, df_ in df.groupby(0):
                TTs = np.mean(store.two_thetas[df_.index], axis=0)
                Is = np.mean(store.intensities[df_.index], axis=0)
                all_Is.append(Is)
                all_ds.append(d)
            two_theta_range = (np.min(store.two_thetas), np.max(store.two_thetas))
            wavelength = store.effective_wavelength
            # Plot the image with all the distance plots
        all_Is = np.array(all_Is)
        all_ds = np.array(all_ds)
        q_range = twotheta_to_q(two_theta_range, wavelength=wavelength)
        # Translate the image to true x,y coordinates via interpolation
        all_ys = all_ds / np.max(all_ds) * (all_Is.shape[0]-1)
        xx, yy = np.meshgrid(np.arange(all_Is.shape[1]), all_ys)
        grid_x, grid_y = np.meshgrid(np.arange(all_Is.shape[1]), np.arange(all_Is.shape[0]))
        points = np.stack([xx, yy], axis=2).reshape((-1, 2))
        new_Is = all_Is
        new_Is = scipy.interpolate.griddata(points, all_Is.flatten(), (grid_x, grid_y))
        # Plot single spectrum closest to a point, for debugging
        # close_idx = np.argmin((all_ds - 1)**2)
        # im = ax.plot(np.linspace(*q_range, num=all_Is.shape[1]), all_Is[close_idx])
        # Plot the actual image
        extent = (*q_range, np.min(r), np.max(r))
        im = ax.imshow(new_Is, aspect='auto', origin='bottom',
                       extent=extent, interpolation='bicubic', *args, **kwargs)
        # Format the axes
        ax.set_xlabel(r'Scattering Length')
        ax.set_ylabel('Distance /mm')
        return im
    
    def plot_scatter(self, metric0: str, metric1: str,
                     weight: str=None, weight_range=(None, None),
                     ax=None, phase_idx=0, **kwargs):
        """Plot two metrics against each other.
        
        Parameters
        ==========
        metric0 
          Metric to plot along the x-axis.
        metric1
          Metric to plot along the y-axis.
        weight
          Metric to use for the sizes of points.
        weight_range : 2-tuple
          (min, max) range for sizes of scatter points.
        ax
          Matplotlib Axes for receiving the plot.
        phase_idx : int
          Index of the crystallographic phase to use. Passed on to
          ``self.metric()``
        **kwargs
          Passed on to matplotlib.scatter
        
        Return
        ------
        artist
          The scatter plot artist.
        
        """
        # Get a new axes if necessary
        if ax is None:
            ax = new_axes()
        # Retrieve the data
        x = self.metric(metric0, phase_idx=phase_idx)
        y = self.metric(metric1, phase_idx=phase_idx)
        s = self.metric(weight, phase_idx=phase_idx)
        # Normalize the particle sizes
        weight_range = (
            weight_range[0] if weight_range[0] is not None else np.min(s),
            weight_range[1] if weight_range[1] is not None else np.max(s),
        )
        s_norm = colors.Normalize(weight_range[0], weight_range[1], clip=True)
        s = s_norm(s)
        s *= rcParams['lines.markersize']**2
        # Plot the scatter
        artist = ax.scatter(x, y, s=s, **kwargs)
        # Decorate the axes
        ax.set_xlabel(metric0)
        ax.set_ylabel(metric1)
        return artist
    
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
    
    def plot_histogram(self, metric: str, ax=None, bins: int=0,
                       weight: str=None, metric_range=None,
                       weight_range=(None, None), cmap=None):
        """Plot a histogram showing the distribution of the given metric.
        
        Arguments
        ---------
        metric :
          String describing which metric to plot. See self.metric()
          for valid choices.
        ax : optional
          Matplotlib axes on which to plot.
        bins : optional
          Number of bins in which to distribute the metric
          values. If zero, the number of bins will be determined
          automatically from the number of loci.
        metric_range : 2-tuple, optional
          Will be used to normalize the range of values for the given
          metric. If omitted, the 2nd and 98th percentile will be
          used.
        weight : String describing which metric to use for weighting
          each value. See self.metric() for valid choices. If None,
          all weights will be equal.
        weight_range : 2-tuple
          Will be used to normalize the values between 1 and 0. If
          not given, then the full range of values will be used.
        cmap : str, optional
          Matplotlib colormap string for converting numeric values
          into colors.
        
        """
        metrics = self.metric(metric)
        # Set default mapping range if not given
        if metric_range is None:
            metric_range = (np.percentile(metrics, 2),
                            np.percentile(metrics, 98))
        # Normalize the data to between 0 and 1
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
            weightnorm = colors.Normalize(*weight_range, clip=True)
            weights = weightnorm(weights)
        if ax is None:
            ax = new_axes(height=4, width=7)
        # Remove nan values
        invalid = np.logical_or(np.isnan(metrics), np.isnan(weights))
        metrics = metrics[~invalid]
        weights = weights[~invalid]
        # Generate the histogram
        n, bins, patches = ax.hist(metrics, bins=bins, weights=weights)
        # Set the colors based on the metric normalizer
        for patch in patches:
            x_position = patch.get_x()
            cmap_ = self.get_cmap(cmap, metric=metric)
            color = cmap_(metricnorm(x_position))
            patch.set_color(color)
        ax.set_xlim(metricnorm.vmin, metricnorm.vmax)
        ax.set_xlabel(metric.replace('_', ' '))
        ax.set_ylabel('Occurrences')
        set_outside_ticks(ax=ax)
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
    Background_Phases = []
    metric_re = re.compile(r'(?P<param>[-_a-zA-Z]+)_(?P<phase>-?\d+)')
    
    def __init__(self, *args, collimator=0.5, qrange=None,
                 scan_time=None, detector_distance=20,
                 frame_size=1024, Phases=[], phases=None,
                 Background_Phases=[], **kwargs):
        # Old-style mapping format deprecation
        if phases is not None:
            warnings.warn(DeprecationWarning(), "Use 'Phases=' instead")
            Phases = phases
        self.collimator = collimator
        self.detector_distance = detector_distance
        self.frame_size = frame_size
        # Checking for nonos-default lists allows for better subclassing
        if len(Phases) > 0:
            self.Phases = Phases
        if len(Background_Phases) > 0:
            self.Background_Phases = Background_Phases
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
    
    def diffractogram(self, index=None, weights=None,
                      weight_range=None, as_twotheta=False, dataset_name='intensities'):
        """Calculate the bulk diffractogram by averaging each scan weighted
        by reliability.
        
	Parameters
	----------
	index : int, optional
	  Which diffractogram to return. If ``None``, 
	  all the diffractograms will be average together.
        weights : str, optional
	  A dataset name to use for weighting each diffractogram.
	  Only used if ``index`` is None.
        weight_range : tuple, optional
          Set the limits of weights such that (min, max).
          If None, weights would be applied without normalization.
        as_twotheta : bool, optional
          If truthy, return the diffractogram with 2θ°. Otherwise, use
          scattering lengths.
        dataset_name : str, optional
          The name of the TXMStore dataset to use for the y-values.
        
        Returns
        =======
        A pandas series of intensity vs scattering length

        """
        with self.store() as store:
            if index is None:
                # First get data from disk
                Is = getattr(store, dataset_name)
                twotheta = np.mean(store.two_thetas, axis=0)
                # Prepare the matrix of the weights for each scan
                if weights is None:
                    Ws = np.ones(shape=(Is.shape[0],))
                else:
                    Ws = self.metric(param=weights)
                # Prepare the normalizer for the weights
                if weight_range is None:
                    norm = lambda x: x
                else:
                    norm = colors.Normalize(vmin=weight_range[0], vmax=weight_range[1], clip=True)
                # Apply the weights to the data
                Ws = np.expand_dims(norm(Ws), axis=-1)
                Is = np.sum(Ws*Is, axis=0)/np.sum(Ws, axis=0)
            else:
                Is = getattr(store, dataset_name)[index]
                twotheta = store.two_thetas[index]
            # Convert to scattering lengths unless 2θ is requested
            if as_twotheta:
                x = twotheta
            else:
                x = twotheta_to_q(twotheta, wavelength=store.effective_wavelength)
            series = pd.Series(Is, index=x)
        return series
    
    def plot_diffractogram(self, ax=None, index=None, subtracted=False, **kwargs):
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
            xs = get_values(store.two_thetas, index=index)
            try:
                bg = get_values(store.backgrounds, index=index)
            except KeyError:
                bg = np.zeros_like(intensities)
            try:
                fits = get_values(store.fits, index=index)
            except KeyError:
                fits = np.zeros_like(intensities)
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
        if subtracted is False:
            ax.plot(xs, observations, marker="+", linestyle="None")
            ax.plot(xs, predictions)
            ax.plot(xs, bg, color="red")
            ax.plot(xs, residuals, color="cyan")
            # Annotate axes
            ax.set_xlabel(r'2θ°')
            ax.set_ylabel('Intensity a.u.')
        else:
            ax.plot(qs, observations, marker="+", linestyle="None")
        if index is not None:
            ax.set_title("Diffractogram for location {idx}".format(idx=index))
        else:
            ax.set_title('Bulk diffractogram')
        # Highlight peaks
        #if substracted:
        #ax.legend("Observations")
        #else:
        color_list = [
            'green',
            'blue',
            'red',
            'orange'
        ]
        # Add a legend
        ax.legend(['Observations', 'Predicted', 'bg', 'residuals'])
        # Helper function to highlight peaks
        def draw_peaks(ax, phase, color):
            """Highlight the expected peak corresponding to this phase."""
            alpha = 0.15
             #Highlight each peak in this phase
            for reflection in phase.reflection_list:
                two_theta = reflection.qrange
                ax.axvspan(two_theta[0], two_theta[1],
                           color=color, alpha=alpha)
        # Highlight phases
        #for idx, phase in enumerate(self.Phases):
            #draw_peaks(ax=ax, phase=phase, color=color_list[idx])
        # Highlight background
        #for phase in self.Background_Phases:
            #draw_peaks(ax=ax, phase=phase, color='grey')
        # Set axes limits
        ax.set_xlim(xs.min(), xs.max())
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
            # diameter = store.step_size.num
            diameter = store.step_size
            loc = xycoord(*store.positions[locus,:])
        diameter = diameter / units.mm
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
    
    def plot_all_diffractograms_2D(self, ax=None, subtracted=False, aspect="auto",
                                   *args, **kwargs):
        """Plot the individual locus diffractograms as an image."""
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
            # Determine dimensions of the image axes
            extent = (np.min(xs), np.max(xs), len(ys)-1, 0)
            # Generate the image plot
            ax.imshow(ys, aspect=aspect, extent=extent, *args, **kwargs)
        return ax
    
    def plot_phase_ratio(self, phase_idx=0, *args, **kwargs):
        warnings.warn(UserWarning("Use `Map.plot_map(metric='phase_ratio')` instead"))
    
    def plot_cell_parameter(self, parameter='a', phase_idx=0, *args, **kwargs):
        warnings.warn(UserWarning(
            "Use `Map.plot_map(metric='{}')` instead".format(parameter)
        ))
    
    def plot_fwhm(self, phase_idx=0, *args, **kwargs):
        warnings.warn(UserWarning("Use `Map.plot_map(metric='fwhm')` instead"))
    
    def refine_mapping_data(self, backend='native', num_bg_coeffs=3, debug_mode: bool=False):
        """Refine the relevant XRD parameters, such as background, unit-cells,
        etc. This will save the refined data to the HDF file.
        
        Different backends are available, namely "native", "fullprof"
        or "pawley". Custom backends may be provided as a subclass of
        :py:class:`scimap.base_refinement.BaseRefinement`.
        
        Parameters
        ----------
        backend : str, Refinement
          The style of refinement to perform. The default 'native'
          uses built-in numerical computations. Other methods will be
          added in the future. A user-created subclass of
          ``BaseRefinement`` can also be supplied.
        num_bg_coeffs : int, optional
          Numbers of background coefficients used for refinement
        debug_mode
          If truthy, failed refinements will provide additional
          information about what went wrong.
        
        """
        # Empty arrays to hold results
        bgs = []
        fits = []
        all_cells = []
        failed = []
        goodness = []
        fractions = []
        scale_factors = []
        broadenings = []
        # Retrieve the refinement method based on the given parameter
        backends = {
            'native': NativeRefinement,
            'pawley': PawleyRefinement,
            'fullprof': FullprofRefinement,
        }
        if callable(backend):
            Refinement = backend
        elif backend in backends.keys():
            Refinement = backends[backend]
        else:
            # Invalid refinement backend, throw exception
            raise exceptions.RefinementError(
                "Invalid backend ``{given}``. Must be one of {strings} "
                "or subclass of ``BaseRefinement``."
                "".format(given=backend, strings=backends.keys()))
        # Open the data storage and start the refining
        with self.store(mode='r+') as store:
            wavelengths = store.wavelengths
            scans = zip(store.two_thetas[:], store.intensities[:])
            total = len(store.two_thetas)
            for idx, (two_theta, Is) in enumerate(prog(scans, total=total, desc="Refining")):
                phases = [P() for P in self.Phases]
                bg_phases = [P() for P in self.Background_Phases]
                file_root = self.sample_name + ('_refinements/locus_%05d_ref' % idx)
                refinement = Refinement(phases=phases, background_phases=bg_phases,
                                        wavelengths=wavelengths, file_root=file_root, 
                                        num_bg_coeffs=num_bg_coeffs)
                try:
                    # Refine background
                    bg = refinement.background(two_theta=two_theta,
                                               intensities=Is)
                    # Refine unit-cell parameters
                    cell_params = refinement.cell_params(
                            two_theta=two_theta,
                            intensities=Is,
                    )
                    # Refine the phase fractions
                    frac = refinement.phase_fractions(
                        two_theta=two_theta,
                        intensities=Is,
                    )
                    # Refine scale factors
                    scale = refinement.scale_factor(
                        two_theta=two_theta,
                        intensities=Is,
                    )
                     #Fit peak widths
                    width = refinement.broadenings(
                        two_theta=two_theta,
                        intensities=Is,
                    )
                    # Append the fitted diffraction pattern
                    fit = refinement.predict(two_theta=two_theta, intensities=Is)
                     #Save a confidence value for the fit
                    gd = refinement.goodness_of_fit(two_theta=two_theta,
                                                    intensities=Is)
                except exceptions.RefinementError as e:
                    # Try plotting the failed refinement
                    if debug_mode:
                        pyplot.figure()
                        pyplot.plot(two_theta, Is)
                        pyplot.title(store.file_basenames[idx])
                        pyplot.show()
                        raise
                    else:
                        failed.append(idx)
                        log.warn('Failed to refine scan {idx}: {e}'.format(idx=idx, e=e))
                        # Refinement was not successful, so save nan values
                        bg = np.full_like(two_theta, np.nan)
                        cell_params = tuple((np.nan,)*6 for p in phases)
                        frac = tuple(np.nan for p in phases)
                        scale = np.nan
                        width = tuple(np.nan for p in phases)
                        fit = np.full_like(two_theta, np.nan)
                        gd = np.nan
                # Added refined values and fitted patterns to cumulative arrays
                bgs.append(bg)
                all_cells.append(cell_params)
                fractions.append(frac)
                scale_factors.append(scale)
                broadenings.append(width)
                goodness.append(gd)
                fits.append(fit)
            # Check that the right number of data were generated
            data_names = [
                ('background', bgs),
                ('cell parameters', all_cells),
                ('fitted patterns', fits),
                ('phase fractions', fractions),
                ('peak broadenings', broadenings),
                ('goodness of fit', goodness),
                ('scale factors', scale_factors),
            ]
            for name, lst in data_names:
                if len(lst) != total:
                    raise exceptions.RefinementError(
                        "Incorrect number of refinements for "
                        "[{}]. Expected {}, got {}"
                        "".format(name, total, len(lst)))
            # Store refined data for later
            store.backgrounds = np.array(bgs)
            store.cell_parameters = np.array(all_cells)
            store.fits = np.array(fits)
            store.phase_fractions = fractions
            store.peak_broadenings = broadenings
            # Normalize goodnesses of fit values to be between 0 and 1
            max_ = np.nanmax(goodness)
            min_ = np.nanmin(goodness)
            goodness = (max_ - goodness) / (max_ - min_)
            store.goodness_of_fit = np.nan_to_num(goodness)
            store.scale_factor = scale_factors
        # Alert the user of failed refinements
        if failed:
            msg = "Could not refine unit cell for loci: {}".format(failed)
            warnings.warn(msg, RuntimeWarning)
    
    def valid_metrics(self):
        """Return a list of the available metrics that a user can map. See
        XRDMap.metric() docstring for a full explanation.
        """
        valid = ['a', 'b', 'c', 'alpha', 'beta', 'gamma', 'integral',
                 'phase_fraction', 'goodness_of_fit', 'position',
                 'peak_broadening', 'scale_factor', 'None']
        return valid
    
    def metric(self, param: str, phase_idx: int=None, locus: int=None):
        """Retrieve a calculated value for each mapping locus.
        
        The phase can be specified in one of two ways. Either 1)
        explicitely by using the ``phase_idx`` parameter or 2) within
        the parameter string like "phase_fraction_1". The explicit
        option takes precedence. If neither is specified, then the
        first phase will be used by default.
        
        **Valid Parameters:**
        - Unit cell parameters: 'a', 'b', 'c', 'alpha', 'beta', 'gamma'
        - 'integral' to indicate total integrated signal after bg subtraction
        - 'goodness_of_fit' to use the quality of fit determined during refinement
        - 'position' to give the distance from the origin (for testing purposes)
        - 'phase_fraction' to give the fraction of the given ``phase_idx``
        - 'None' or None to give an array of 1's
        
        Parameters
        ==========
        param
          Name of the parameter to be retrieved, as described above.
        phase_idx : optional
          Which crystallographic phase to use, default is 0. See above
          for more detailed behavior.
        locus : optional
          Index of mapping locus (position) to retrieve. Result will
          still be an array, but with only one value.
        
        Returns
        -------
        metric
          A numpy array with the requested metrics. If `locus` is
          None, this array will contain all loci, otherwise it will be
          only the requested locus.
        
        """
        # Extract the phase_idx if necessary
        phase_match = self.metric_re.match(param)
        if phase_idx is not None and phase_match:
            raise AttributeError("Don't provide phase index as both ``param='%s'``"
                                 "and ``phase_idx=%d``." % (param, phase_idx))
        elif phase_match:
            param = phase_match.group('param')
            phase_idx = int(phase_match.group('phase'))
        else:
            phase_idx = 0
        # Check for unit-cell parameters
        UNIT_CELL_PARAMS = ['a', 'b', 'c', 'alpha', 'beta', 'gamma']
        if param in UNIT_CELL_PARAMS:
            param_idx = UNIT_CELL_PARAMS.index(param)
            with self.store() as store:
                metric = store.cell_parameters[:,phase_idx,param_idx]
        elif param == 'integral':
            with self.store() as store:
                xs = store.two_thetas
                I = store.intensities.value - store.backgrounds.value
                metric = scipy.integrate.trapz(y=I, x=xs, axis=1)
        elif param == "position":
            with self.store() as store:
                locs = store.positions
                metric = np.sqrt(locs[:,0]**2 + locs[:,1]**2)
        elif param in ['goodness_of_fit', 'scale_factor']:
            # Just return the requested store attribute
            with self.store() as store:
                metric = getattr(store, param).value
        elif param == 'phase_fraction':
            with self.store() as store:
                metric = store.phase_fractions[:,phase_idx]
        elif param == 'peak_broadening':
            with self.store() as store:
                metric = store.peak_broadenings[:,phase_idx]
        elif param in ['None', None]:
            # Return a dummy array with all 1's
            with self.store() as store:
                metric = np.ones(shape=store.scattering_lengths.shape[0])
        else:
            raise ValueError("Unknown param: {}. Options are: {}"
                             "".format(param, self.valid_metrics()))
        # Filter by requested locus
        if locus is not None:
            metric = metric[locus:locus+1]
        return metric
    
    def plot_map_gtk(self, *args, **kwargs):
        from .gtkmapviewer import GtkMapViewer
        return super().plot_map_gtk(WindowClass=GtkMapViewer, *args, **kwargs)
    
    def dots_per_mm(self):
        """Determine the width of the scan images based on sample's camera
        zoom (dpm taken from camera calibration using quadratic
        regression on Bruker D8 Discover Series II)
        """
        def regression(x):
            return 3.640 * x**2 + 13.869 * x + 31.499
        dots_per_mm = regression(self.camera_zoom)
        return dots_per_mm
    
    def prepare_mapping_data(self, *args, **kwargs):
        raise exceptions.DeprecationError("Use `Map.refine_mapping_data() instead`")
    
    def refine_scans(self):
        raise exceptions.DeprecationError("Use `Map.refine_mapping_data() instead`")
    
    def plot_cell_parameter(self, parameter='a', phase_idx=0, *args, **kwargs):
        raise exceptions.DeprecationError("Use `Map.plot_map(metric='{}')` instead"
                                          "".format(parameter))
    
    def plot_fwhm(self, phase_idx=0, *args, **kwargs):
        raise exceptions.DeprecationError("Use `Map.plot_map(metric='fwhm')` instead")
    
    def plot_phase_ratio(self, phase_idx=0, *args, **kwargs):
        raise exceptions.DeprecationError("Use `Map.plot_map(metric='phase_ratio')` instead")
    
    def write_script(self, file=None, quiet=False):
        raise exceptions.DeprecationError("Use gadds.write_gadds_script()")
