# -*- coding: utf-8 --*

import warnings

from matplotlib import patches
import numpy as np
import scipy
import pandas

from .. import exceptions
from .. mapping.map import Map
from .locus import XRDLocus
from .xrdstore import XRDStore
from ..refinement.native import NativeRefinement
from ..utilities import prog, xycoord


class XRDMap(Map):
    """A map using X-ray diffraction to determine cell values. Runs on
    Bruker D8 Discover using GADDS software. Collimator size
    determines resolution in mm. `scan_time` directs how long the
    instrument spends at each point (in seconds).

    The parameter 'phases' is a list of *uninitialized* Phase
    classes. These will be initialized separately for each scan.

    """
    locus_class = XRDLocus
    cell_parameter_normalizer = None
    phase_ratio_normalizer = None
    fwhm_normalizer = None
    THETA1_MIN = 0 # Source limits based on geometry
    THETA1_MAX = 50
    THETA2_MIN = 0 # Detector limits based on geometry
    THETA2_MAX = 55
    camera_zoom = 6
    two_theta_range = (10, 80)
    frame_step = 20  # How much to move detector by in degrees
    frame_width = 20  # 2-theta coverage of detector face
    scan_time = 300  # In seconds
    Phases = []
    background_phases = []

    def __init__(self, *args, collimator=0.5, qrange=None,
                 scan_time=None, detector_distance=20,
                 frame_size=1024, Phases=[], phases=None,
                 background_phases=[],
                 refinement=NativeRefinement, **kwargs):
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
        self.refinement = refinement
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

    def new_locus(self, *, location, filebase):
        """Create a new XRD Mapping cell with the given attributes as well as
        associated crystallographic phases."""
        # Initialize list of crystallographic phases
        phases = [Phase() for Phase in self.phases]
        background_phases = [Phase() for Phase in self.background_phases]
        # Create mapping locus
        new_locus = XRDLocus(location=location, parent_map=self, filebase=filebase,
                             phases=phases, background_phases=background_phases,
                             two_theta_range=self.two_theta_range,
                             refinement=self.refinement)
        return new_locus

    def write_script(self, file=None, quiet=False):
        """
        Format the sample into a slam file that GADDS can process.
        """
        raise NotImplementedError("Use gadds.write_gadds_script()")

    @property
    def diffractogram(self):
        """Returns self.bulk_diffractogram(). Polymorphism for XRDScan."""
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

    def refine_mapping_data(self):
        """Refine the relevant XRD parameters, such as background, unit-cells,
        etc. This will save the refined data to the HDF file."""
        bgs = []
        fits = []
        all_cells = []
        failed = []
        goodness = []
        with self.store(mode='r+') as store:
            scans = zip(store.scattering_lengths[:], store.intensities[:])
            total = len(store.scattering_lengths)
            for idx, (qs, Is) in enumerate(prog(scans, total=total)):
                phases = [P() for P in self.Phases]
                refinement = self.refinement(phases=phases)
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
                fits.append(np.zeros_like(qs))
                # fits.append(refinement.predict(qs))
            # Store refined data for later
            store.backgrounds = np.array(bgs)
            store.cell_parameters = np.array(all_cells)
            store.fits = np.array(fits)
            # Normalize goodnesses of fit values to be between 0 and 1
            max_ = np.nanmax(goodness)
            min_ = np.nanmin(goodness)
            goodness = (max_ - goodness) / (max_ - min_)
            store.goodness = np.nan_to_num(goodness)
        # Alert the user of failed refinements
        if failed:
            msg = "Could not refine unit cell for loci: {}".format(failed)
            warnings.warn(msg, RuntimeWarning)

    def set_metric_phase_ratio(self, phase_idx=0):
        """Set the plotting metric as the proportion of given phase."""
        for locus in prog(self.loci, desc='Calculating metrics'):
            phase_scale = locus.phases[phase_idx].scale_factor
            total_scale = sum([phase.scale_factor for phase in locus.phases])
            locus.metric = phase_scale / total_scale

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
        """Plot a map of the ratio of the given phase index to all the phases"""
        self.set_metric_phase_ratio(phase_idx=0)
        self.metric_name = 'Phase ratio'
        # Determine normalization range
        if self.phase_ratio_normalizer is None:
            self.metric_normalizer = self.fullrange_normalizer()
            self.calculate_normalizer()
        else:
            self.metric_normalizer = self.phase_ratio_normalizer
        # Plot the map
        ax = self.plot_map(*args, **kwargs)
        return ax

    def set_metric_cell_parameter(self, parameter='a', phase_idx=0):
        for locus in prog(self.loci, desc='Calculating cell parameters'):
            phase = locus.phases[phase_idx]
            locus.metric = getattr(phase.unit_cell, parameter)

    def plot_cell_parameter(self, parameter='a', phase_idx=0, *args, **kwargs):
        self.set_metric_cell_parameter(parameter, phase_idx)
        self.metric_name = 'Unit-cell parameter {0} Å'.format(parameter)
        # Determine normalization range
        if self.cell_parameter_normalizer is None:
            self.metric_normalizer = self.fullrange_normalizer()
        else:
            self.metric_normalizer = self.cell_parameter_normalizer
        # Now plot the map
        return self.plot_map(*args, **kwargs)

    def set_metric_fwhm(self, phase_idx=0):
        for locus in prog(self.loci, desc='Calculating peak widths'):
            locus.metric = locus.refinement.fwhm()

    def plot_fwhm(self, phase_idx=0, *args, **kwargs):
        self.set_metric_fwhm(phase_idx=phase_idx)
        self.metric_name = 'Full-width half max (°)'
        # Determine normalization range
        if self.fwhm_normalizer is None:
            self.metric_normalizer = self.fullrange_normalizer()
        else:
            self.metric_normalizer = self.fwhm_normalizer
        # Now plot the map
        return self.plot_map(*args, **kwargs)

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
        warnings.warn(DeprecationWarning("Use `xrd.imports.import_gadds_map`"))
        # for locus in self.loci:
        #     locus.load_diffractogram()
        # self.refine_scans()
        # return super().prepare_mapping_data()

    def refine_scans(self):
        """
        Refine a series of parameters on each scan. Continue if an
        exceptions.RefinementError occurs.
        """
        for locus in prog(self.loci, desc='Reticulating splines'):
            try:
                current_step = 'background'
                locus.refinement.refine_background()
                current_step = 'displacement'
                locus.refinement.refine_displacement()
                current_step = 'peak_widths'
                locus.refinement.refine_peak_widths()
                current_step = 'unit cells'
                locus.refinement.refine_unit_cells()
                # if len(locus.phases) > 2:
                current_step = 'scale factors'
                locus.refinement.refine_scale_factors()
            except exceptions.SingularMatrixError as e:
                # Display an error message on exception and then coninue fitting
                msg = "{coords}: {msg}".format(coords=locus.cube_coords, msg=e)
                print(msg)
            except exceptions.DivergenceError as e:
                msg = "{coords}: DivergenceError while refining {step}".format(
                    coords=locus.cube_coords,
                    step=current_step
                )
                print(msg)
            except exceptions.PCRFileError as e:
                msg = "Could not read resulting pcr file: {}".format(e)
                print(msg)
