# -*- coding: utf-8 --*

import math
import os

import numpy as np
import pandas
import jinja2

import exceptions
from mapping.map import Map
from plots import new_axes, DegreeFormatter
from .locus import XRDLocus
from .xrdstore import XRDStore
from refinement.native import NativeRefinement
from utilities import prog


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
    phases = []
    background_phases = []

    def __init__(self, *args, collimator=0.5, two_theta_range=None,
                 scan_time=None, detector_distance=20,
                 frame_size=1024, phases=[], background_phases=[],
                 refinement=NativeRefinement, **kwargs):
        self.collimator = collimator
        self.detector_distance = detector_distance
        self.frame_size = frame_size
        # Checking for non-default lists allows for better subclassing
        if len(phases) > 0:
            self.phases = phases
        if len(background_phases) > 0:
            self.background_phases = background_phases
        if scan_time is not None:
            self.scan_time = scan_time
        self.refinement = refinement
        if two_theta_range is not None:
            self.two_theta_range = two_theta_range
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
        # Estimate the total time
        totalSecs = len(self.loci) * self.scan_time * self.get_number_of_frames()
        days = math.floor(totalSecs / 60 / 60 / 24)
        remainder = totalSecs - days * 60 * 60 * 24
        hours = math.floor(remainder / 60 / 60)
        remainder = remainder - hours * 60 * 60
        mins = math.floor(remainder / 60)
        total_time = "{secs}s ({days}d {hours}h {mins}m)".format(secs=totalSecs,
                                                                 days=days,
                                                                 hours=hours,
                                                                 mins=mins)
        # List of frames to integrate
        frames = []
        for frame_num in range(0, self.get_number_of_frames()):
            start = self.two_theta_range[0] + frame_num * self.frame_step
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
            'num_scans': len(self.loci),
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
        for idx, locus in enumerate(self.loci):
            # Prepare scan-specific details
            x, y = locus.xy_coords(unit_size=self.unit_size)
            scan_metadata = {'x': x, 'y': y, 'filename': locus.filebase}
            context['scans'].append(scan_metadata)
        return context

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
        # Import template
        env = jinja2.Environment(loader=jinja2.PackageLoader('scimap', ''))
        template = env.get_template('mapping/mapping-template.slm')
        self.create_loci()
        context = self.context()
        # Create file and directory if necessary
        if file is None:
            directory = self.directory()
            if not os.path.exists(directory):
                os.makedirs(directory)
            filename = '{dir}/{samplename}.slm'.format(
                dir=directory, samplename=self.sample_name
            )
            with open(filename, 'w') as file:
                file.write(template.render(**context))
        else:
            file.write(template.render(**context))
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
        return file

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

    def plot_diffractogram(self, ax=None, index=None):
        """Plot a specific diffractogram or an average of all the scans,
        weighted by reliability.

        Arguments
        ---------
        - ax : The matplotlib axes object to plot on to

        - index : Which locus to plot.
        """
        def get_values(data, index):
            if index is None:
                return np.average(data, axis=0)
            else:
                return data[index]
        with self.store() as store:
            intensities = get_values(store.intensities, index=index)
            qs = get_values(store.scattering_lengths, index=index)
            bg = get_values(store.backgrounds, index=index)
        # Get default axis if none is given
        if ax is None:
            ax = new_axes()
        ax.plot(qs, intensities)
        print(qs.shape, bg.shape)
        ax.plot(qs, bg)
        ax.set_xlabel(r'$2\theta$')
        ax.set_ylabel('counts')
        ax.set_title('Bulk diffractogram')
        ax.xaxis.set_major_formatter(DegreeFormatter())
        # Highlight peaks
        # self.loci[0].xrdscan.refinement.highlight_peaks(ax=ax)
        return ax

    def refine_mapping_data(self):
        """Refine the relevant XRD parameters, such as background, unit-cells,
        etc."""
        refinement = self.refinement(phases=self.phases)
        bgs = []
        with self.store(mode='r+') as store:
            for qs, intensities in zip(store.scattering_lengths, store.intensities):
                bg = refinement.refine_background(scattering_lengths=qs,
                                                  intensities=intensities)
                bgs.append(bg)
            store.backgrounds = bgs

    def set_metric_phase_ratio(self, phase_idx=0):
        """Set the plotting metric as the proportion of given phase."""
        for locus in prog(self.loci, desc='Calculating metrics'):
            phase_scale = locus.phases[phase_idx].scale_factor
            total_scale = sum([phase.scale_factor for phase in locus.phases])
            locus.metric = phase_scale / total_scale

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

    def plot_map_gtk(self):
        from xrd.gtkmapviewer import GtkXrdMapViewer
        return super().plot_map_gtk(WindowClass=GtkXrdMapViewer)

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
        for locus in self.loci:
            locus.load_diffractogram()
        self.refine_scans()
        return super().prepare_mapping_data()

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
