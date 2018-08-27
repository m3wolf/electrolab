# -*- coding: utf-8 -*-

from enum import Enum
import math
import os
import re
from subprocess import call
import contextlib
import logging

log = logging.getLogger(__name__)

import jinja2
import pandas as pd
import numpy as np
from matplotlib.colors import Normalize
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d, CubicSpline

from . import plots
from . import exceptions
from .base_refinement import BaseRefinement
from .phase import Phase


def fp2k():
    """Determine the path to the fullprof binary.
    
    Returns
    -------
    cmd : str
      Absolute path to the Fullprof binary.
    
    Raises
    ------
    RefinementError
      If the Fullprof binary cannot be found.
    
    """
    dir_ = os.environ.get('FULLPROF', os.getcwd())
    cmd = os.path.join(dir_, 'fp2k')
    if not os.path.exists(cmd):
        raise exceptions.RefinementError(
            "Could not find Fullprof binary. Make sure the $FULLPROF "
            "environemtal variable is set to the Fullprof directory."
        )
    return cmd


class FailPlotter():
    def __init__(self, file_root):
        self.file_root = file_root
    
    def __enter__(self):
        self.dataframes = []
        self.descriptions = []
        return self
    
    def __exit__(self, e_type, e_value, e_tb):
        self.plot_dataframes()
        if e_type is not None:
            raise
    
    def add_dataframe(self, dataframe, description=''):
        self.dataframes.append(dataframe)
        self.descriptions.append(description)
    
    def plot_dataframes(self):
        # Prepare an array of plots for the results
        n_rows = math.ceil(len(self.dataframes) / 2)
        panel_size = 5
        if n_rows > 0:
            figsize = (2 * panel_size, n_rows * panel_size)
            fig, axs = plt.subplots(n_rows, 2, figsize=figsize, squeeze=False)
            axs = tuple(a for ax in axs for a in ax)
            # Now plot the result of each scan on the appropriate axes
            for idx, (df, desc, ax) in enumerate(zip(self.dataframes, self.descriptions, axs)):
                twotheta = df[' 2Theta']
                ax.plot(twotheta, df['Yobs'], marker='+', linestyle="None")
                ax.plot(twotheta, df['Ycal'])
                ax.plot(twotheta, df['Backg'], color='red')
                ax.plot(twotheta, df['Yobs-Ycal'], color='cyan')
                ax.legend(['Obs', 'Calc', 'Bg', 'Diff'])
                # Add some metadata
                ax.set_title(desc)
                ax.set_xlabel('2θ°')
                ax.set_ylabel('Intensity')
                ax.spines['right'].set_visible(False)
                ax.spines['top'].set_visible(False)
            # Remove leftover panel if necessary
            for idx in range(-1, -1-(2*n_rows-len(self.dataframes)), -1):
                axs[idx].remove()
            # Save the new plots to disk
            pdf_fname = self.file_root + '.pdf'
            plt.tight_layout()
            fig.savefig(pdf_fname)
        else:
            log.info("No successful refinements to plot for {}".format(self.file_root))


def is_number(s):
    # Check for invalid, but still numerical, values
    if s.lower() in ['nan', 'inf']:
        return False
    # Check that it's actually a number
    try:
        float(s)
        return True
    except ValueError:
        return False


@contextlib.contextmanager
def read_pcr_file(section_name=None):
    """Context manager handles exceptions when PCR file is malformed."""
    try:
        yield
    except AttributeError:
        raise exceptions.RefinementError("Invalid ")


def load_refined_diffractogram(filename):
    return pd.read_csv(filename, skiprows=3, sep='\t')


def save_diffractogram(filename, two_theta, intensities):
    """Save data as a .dat file format suitable for Fullprof input.
    
    Parameters
    ----------
    filename : str
      Filename for the resulting data file.
    two_theta : np.ndarray
      Diffraction angles in 2θ°.
    intensities : np.ndarray
      Observed diffraction intensities at the given 2θ values.
    
    """
    series = pd.Series(intensities, index=two_theta)
    series.to_csv(filename, sep=' ', header=False)


def plot_refinement(filename, ax=None):
    if ax is None:
        ax = plots.new_axes()
    df = load_refined_diffractogram(filename=filename)
    ax.plot(df[' 2Theta'], df['Yobs'])
    ax.plot(df[' 2Theta'], df['Ycal'])
    ax.plot(df[' 2Theta'], df['Yobs-Ycal'])
    ax.set_title('Profile refinement {filename}'.format(filename=filename))
    ax.set_xlim(
        right=df[' 2Theta'].max()
    )
    return ax


class FullProfPhase(Phase):
    scale_factor = 1
    isotropic_temp = 0
    # Width parameters
    u = 0.008
    v = -0.004
    w = 0.003
    I_g = 0
    # Shape parameters
    eta = 0.5
    x = 0
    # data_dict = DataDict(['scale_factor', 'isotropic_temp',
    #                       'u', 'v', 'w', 'I_g',
    #                       'eta', 'x'])


class Mode(Enum):
    """
    Refinement modes used by fullprof for the Jbt value.
    """
    rietveld = 0
    magnetic = 1
    constant_scale = 2
    constant_intensities = 3


class FullprofRefinement(BaseRefinement):
    keep_temp_files = False  # delete temp files after refinement?
    bg_coeffs = [0, 0, 0, 0, 0, 0]  # Sixth degree polynomial
    chi_squared = None
    zero = 0  # Instrument non-centrality
    displacement = 0.00032  # cos (θ) dependence
    transparency = -0.00810  # sin (θ) dependence
    # Regular expressions for reading output summary files
    score_re = re.compile(
        '=> Scor:\s*([-0-9Ea.Nan]+)')
    success_re = re.compile(
        '==> RESULTS OF REFINEMENT:')
    chi_re = re.compile(
        'Chi2:\s+([-0-9Ee.Na]+)')
    bg_re = re.compile(
        'Background Polynomial Parameters ==>((?:\s+[-+0-9Ee.]+)+)')
    displacement_re = re.compile(
        'Cos\( theta\)-shift parameter :\s+([-0-9Ee.]+)')
    scale_re = re.compile(
        '=> overall scale factor :\s+([-0-9Ee.]+)\s+([-0-9Ee.]+)')
    width_re = re.compile(
        '=> Halfwidth parameters\s+:\s+((?:\s+[-0-9Ee.]+)+)')
    cell_re = re.compile(
        '=> Cell parameters\s+:\s+((?:\s+[-0-9Ee.]+)+)')
    # Regular expressions for reading errors from log file
    singular_matrix_re = re.compile(
        '==> Singular matrix!!, problem with (\S+)')
    divergence_re = re.compile(
        '=>  Unrecoverable divergence!!')
    negative_fwhm_re = re.compile(
        '=> Negative GAUSSIAN and/or LORENTZIAN FWHM detected')
    
    def __init__(self, num_bg_coeffs=3, *args, **kwargs):
        if not (0 <= num_bg_coeffs <= 6):
            raise ValueError('Value of num_bg_coeffs must be between 0 and 6.')
        self.num_bg_coeffs = num_bg_coeffs
        super().__init__(*args, **kwargs)
        
    @property
    def all_phases(self):
        return self.phases + self.background_phases
   
    @property
    def calculated_diffractogram(self):
        """Read a pcf file and return the refinement as a dataframe."""
        # Check for cached version
        df = load_refined_diffractogram(self.file_root + '.prf')
        return df
    
    def write_hkl_file(self, phase, filename):
        # Write separate hkl file for each phase
        hkl_string = " {h} {k} {l} {mult} {intensity}\n"
        with open(filename, 'w') as hklfile:
            # Write header
            hklfile.write('{}\n'.format(phase))
            hklfile.write('30 0 0.00 SPGr: {}\n'.format(phase.fullprof_spacegroup))
            # Write list of reflections
            for reflection in phase.reflection_list:
                hkl = reflection.hkl
                hklfile.write(hkl_string.format(h=hkl.h, k=hkl.k, l=hkl.l,
                                                mult=reflection.multiplicity,
                                                intensity=reflection.intensity))
    
    def run_fullprof(self, context, two_theta, intensities):
        """Prepare a pcr file and execute the actual fullprof program."""
        # Check that the temporary refinement directory exists
        basedir = os.path.dirname(self.file_root)
        if not os.path.exists(basedir):
            os.makedirs(basedir)
        # Make sure the context is sane
        if context['num_params'] == 0:
            msg = "context['num_params'] is zero"
            raise exceptions.EmptyRefinementError(msg)
        if context['refinement_mode'] == Mode.constant_scale:
            context['Irf'] = 0  # Reflections generated by FullProf
        else:
            context['Irf'] = 2  # Need to save codefile
        # Prepare pcr file
        env = jinja2.Environment(loader=jinja2.PackageLoader('scimap', ''))
        template = env.get_template('templates/fullprof-template.pcr')
        with open(self.pcrfilename, mode='w') as pcrfile:
            pcrfile.write(template.render(**context))
        # Write datafile
        save_diffractogram(self.datafilename, two_theta=two_theta,
                           intensities=intensities)
        # Execute refinement
        with open(self.logfilename, 'w') as logfile:
            call([fp2k(), self.pcrfilename], stdout=logfile)
        # Read refined values
        try:
            self.load_results()
        except exceptions.RefinementError:
            raise
        else:
            # If all went well, load refined pattern and delete temporary files
            self.calculated_diffractogram
    
    def background(self, two_theta, intensities):
        """Retrieve the predicted background array.
        
        If refinement of the background has not been done, this method
        will first perform the refinement process.
        
        Parameters
        ----------
        two_theta : np.ndarray, optional
          An array of 2θ° values.
        intensities : np.ndarray, optional
          Observed diffraction intensities for the 2θ values. Used for
          refinement if necessary.
        
        Returns
        -------
        background : np.ndarray
          The predicted background intensities for the 2θ values.
        
        """
        if not self.is_refined['background']:
            self.refine_all(two_theta, intensities)
        # Load the calculated background
        calc = self.calculated_diffractogram
        bg = self.respline(x=calc[' 2Theta'], y=calc['Backg'],
                           new_x=two_theta)
        return bg
    
    def cell_params(self, two_theta, intensities):
        """Retrieve the predicted cell parameters.
        
        If refinement of the cell parameters has not been done, this
        method will first perform the refinement process.
        
        Parameters
        ----------
        two_theta : np.ndarray, optional
          An array of 2θ° values.
        intensities : np.ndarray, optional
          Observed diffraction intensities for the 2θ values. Used for
          refinement if necessary.
        
        Returns
        -------
        cell_params : tuple
          The predicted cell parameters
        
        """
        if not self.is_refined['unit_cells']:
            self.refine_all(two_theta, intensities)
        # Load the calculated cell params
        params = [p.unit_cell.as_tuple() for p in self.phases]
        return params
    
    def phase_fractions(self, two_theta, intensities):
        """Retrieve the predicted phase fractions.
        
        If refinement of the phase fractions has not been done, this
        method will first perform the refinement process.
        
        Parameters
        ----------
        two_theta : np.ndarray, optional
          An array of 2θ° values.
        intensities : np.ndarray, optional
          Observed diffraction intensities for the 2θ values. Used for
          refinement if necessary.
        
        Returns
        -------
        phase_fractions : tuple
          The predicted phase fractions, 1 for each phase.
        
        """
        if not self.is_refined['scale_factors']:
            self.refine_all(two_theta, intensities)
        # Retrieve refined factors from phases
        scales = tuple(p.scale_factor for p in self.phases)
        phase_fractions = scales / np.sum(scales)
        return phase_fractions
    
    def scale_factor(self, two_theta=None, intensities=None):
        """Retrieve the overall contribution for the whole pattern.
        
        If refinement of the scale_factor has not been done, this
        method will first perform the refinement process.
        
        Parameters
        ----------
        two_theta : np.ndarray, optional
          An array of 2θ° values.
        intensities : np.ndarray, optional
          Observed diffraction intensities for the 2θ values. Used for
          refinement if necessary.
        
        Returns
        -------
        scale_factor : float
          The predicted scale factor for the pattern.
        
        """
        if not self.is_refined['scale_factors']:
            self.refine_all(two_theta, intensities)
        # Retrieve refined factors from phases
        scale_factor = sum(p.scale_factor for p in self.phases)
        return scale_factor
    
    def respline(self, x, y, new_x):
        """Takes a set of xy data and re-interprets it with new X values.
        
        Useful if the output of refinement does not have the same
        dimensions as the input data.
        
        Parameters
        ----------
        x : np.array
          The current x values.
        y : np.array
          The current y values. Must have same shape as ``x``.
        new_x : np.array
          The new x values to use.
        
        Returns
        -------
        new_y : np.array
          The new predicted y values. Will have same shape as ``new_x``.
        
        """
        f = CubicSpline(x, y)
        new_y = f(new_x)
        return new_y
    
    @property
    def logfilename(self):
        return self.file_root + '.log'
    
    @property
    def datafilename(self):
        return self.file_root + '.dat'
    
    @property
    def pcrfilename(self):
        return self.file_root + '.pcr'
    
    def refine_all(self, two_theta, intensities):
        """Execute all the refinements in a reasonable order."""
        # Write an hkl file for each phase
        hkl_filenames = []
        for idx, phase in enumerate(self.all_phases):
            hklfilename = '{}{}.hkl'.format(self.file_root, idx+1)
            hkl_filenames.append(hklfilename)
            self.write_hkl_file(phase, hklfilename)
        with FailPlotter(file_root=self.file_root) as plotter:
            # Refine the background first
            #context = self.pcrfile_context()
            #self.add_background_refinement(context)
            #self.run_fullprof(context=context, two_theta=two_theta, intensities=intensities)
            #self.is_refined['background'] = True
            #plotter.add_dataframe(self.calculated_diffractogram,
             #                     description='Background refined')
            # Refine unit-cell parameters
            context = self.pcrfile_context()
            self.add_background_refinement(context)
            self.add_unit_cell_refinement(context)
            self.run_fullprof(context=context, two_theta=two_theta, intensities=intensities)
            self.is_refined['unit_cells'] = True
            plotter.add_dataframe(self.calculated_diffractogram,
                                  description='Unit cells refined')
            # Refine phase scale factors
            context = self.pcrfile_context()
            self.add_background_refinement(context)
            self.add_unit_cell_refinement(context)
            if len(self.all_phases) > 1:
                self.add_scale_factor_refinement(context)
            self.run_fullprof(context=context, two_theta=two_theta, intensities=intensities)
            self.is_refined['scale_factors'] = True
            plotter.add_dataframe(self.calculated_diffractogram,
                                  description='Scale factors refined')
            # Refine phase peak broadenings
            context = self.pcrfile_context()
            # self.add_background_refinement(context)
            # self.add_unit_cell_refinement(context)
            # if len(self.all_phases) > 1:
                # self.add_scale_factor_refinement(context)
            self.add_peak_widths_refinement(context)
            self.run_fullprof(context=context, two_theta=two_theta, intensities=intensities)
            self.is_refined['peak_widths'] = True
            plotter.add_dataframe(self.calculated_diffractogram,
                                  description='Peak broadening refined')
            # Refine sample displacement
            context = self.pcrfile_context()
            self.add_background_refinement(context)
            self.add_unit_cell_refinement(context)
            # if len(self.all_phases) > 1:
            #     self.add_scale_factor_refinement(context)
            # self.add_peak_widths_refinement(context)
            self.add_displacement_refinement(context)
            self.run_fullprof(context=context, two_theta=two_theta, intensities=intensities)
            self.is_refined['displacement'] = True
            plotter.add_dataframe(self.calculated_diffractogram,
                                  description='Displacement refined')
            # Refine a final pass
            context = self.pcrfile_context()
            self.add_background_refinement(context)
            self.add_unit_cell_refinement(context)
            if len(self.all_phases) > 1:
                self.add_scale_factor_refinement(context)
            self.add_peak_widths_refinement(context)
            self.add_displacement_refinement(context)
            self.run_fullprof(context=context, two_theta=two_theta, intensities=intensities)
            self.is_refined['displacement'] = True
            plotter.add_dataframe(self.calculated_diffractogram,
                                  description='Displacement refined')
        # Remove temporary files
        if not self.keep_temp_files:
            os.remove(self.logfilename)
            os.remove(self.file_root + '.sum')
            os.remove(self.file_root + '.out')
            os.remove(self.datafilename)
            [os.remove(f) for f in hkl_filenames]
            os.remove(self.pcrfilename)
            # os.remove(self.file_root + '.prf')
    
    def add_background_refinement(self, context):
        """
        Refine the six background coefficients.
        """
        coeffs = [11, 21, 31, 41, 51, 61]
        # Mask some coefficients based on input
        for idx in range(self.num_bg_coeffs, len(coeffs)):
            coeffs[idx] = 0
        context['bg_codewords'] = coeffs
        context['num_params'] += self.num_bg_coeffs
        # Refining scale factors simultaneously helps with the fit
        if not context['refinement_mode'] == Mode.constant_scale:
            for idx, phase in enumerate(context['phases']):
                scale_cw = (idx + 4) * 10 + 1
                phase['codewords']['scale'] = scale_cw
    
    def add_displacement_refinement(self, context):
        """
        Refine sample displacement cos θ dependendent correction.
        """
        context['displacement_codeword'] = 11
        context['num_params'] += 1
    
    def add_peak_widths_refinement(self, context):
        context['num_params'] += len(context['phases']) * 1
        for idx, phase in enumerate(context['phases']):
            # phase['codewords']['u'] = (idx*2+1)*10 + 1
            phase['codewords']['w'] = (idx * 1 + 1) * 10 + 1
            # phase['codewords']['v'] = (idx*2+2)*10 + 1
    
    def add_unit_cell_refinement(self, context):
        context['num_params'] += len(context['phases']) * 6
        for idx, phase in enumerate(context['phases']):
            phase['codewords']['a'] = (idx * 6 + 1) * 10 + 1
            phase['codewords']['b'] = (idx * 6 + 2) * 10 + 1
            phase['codewords']['c'] = (idx * 6 + 3) * 10 + 1
            phase['codewords']['alpha'] = (idx * 6 + 4) * 10 + 1
            phase['codewords']['beta'] = (idx * 6 + 5) * 10 + 1
            phase['codewords']['gamma'] = (idx * 6 + 6) * 10 + 1
        return context
    
    def add_scale_factor_refinement(self, context):
        # Refining scale factors for only 1 phases won't work
        if len(context['phases']) < 2:
            msg = 'Cannot refine scale factors with {} phase(s)'
            raise exceptions.RefinementError(
                msg.format(len(context['phases']))
            )
        # Must be in constant intensities mode to refine a scale factor
        context['num_params'] += len(context['phases'])
        for idx, phase in enumerate(context['phases']):
            phase['codewords']['scale'] = (idx + 1) * 10 + 1
    
    def load_results(self, filename=None):
        """
        After a refinement, load the result (.sum) file and restore parameters.
        """
        if filename is None:
            filename = self.file_root + '.sum'
        with open(filename) as summaryfile:
            summary = summaryfile.read()
        # Check for successful refinement
        success = False
        if self.success_re.search(summary):
            score = self.score_re.search(summary).group(1)
            if is_number(score):
                success = True
        if not success:
            # Refinement was not successful, look through log file for clues
            with open(self.file_root + '.log') as logfile:
                log = logfile.read()
            singular_match = self.singular_matrix_re.search(log)
            divergence_match = self.divergence_re.search(log)
            negative_fwhm_match = self.negative_fwhm_re.search(log)
            if singular_match:
                # Singular matrix errors usually means refinement to zero on a given parameter
                param = singular_match.group(1)
                raise exceptions.SingularMatrixError(param=param)
            elif divergence_match:
                # Divergence means the parameter cannot find a minimum
                raise exceptions.DivergenceError()
            else:
                msg = 'Refinement error in scan {}'.format(self.file_root)
                raise exceptions.RefinementError(msg)
        # Search for final Χ² value
        chi_squared = float(self.chi_re.search(summary).group(1))
        if chi_squared == 'NaN':
            raise exceptions.NoReflectionsError()
        self.chi_squared = chi_squared
        # Search for the background coeffs
        bg_results = self.bg_re.search(summary).group(1).split()
        # (Even values are coeffs, odd values are standard deviations)
        bg_coeffs = [float(x) for x in bg_results[::2]]
        # bg_stdevs = [float(x) for x in bg_results[1::2]]
        self.bg_coeffs = bg_coeffs
        # Search for sample displacement correction
        displacement = float(self.displacement_re.search(summary).group(1))
        self.displacement = displacement
        # Search for scale factors
        scale_matches = self.scale_re.findall(summary)
        if len(scale_matches) < len(self.all_phases):
            raise exceptions.RefinementError()
        for idx, phase in enumerate(self.phases):
            match = scale_matches[idx]
            phase.scale_factor = float(match[0])
            phase.scale_error = float(match[1])
        # Search for peak-width parameters
        width_matches = self.width_re.findall(summary)
        if len(width_matches) < len(self.all_phases):
            raise exceptions.RefinementError()
        for idx, phase in enumerate(self.phases):
            match = width_matches[idx].split()
            width_params = [float(x) for x in match[::2]]
            # width_stdevs = [float(x) for x in match[1::2]]
            try:
                phase.u = width_params[0]
                phase.v = width_params[1]
                phase.w = width_params[2]
            except IndexError:
                # Index error usually means some sort of ******* parameter
                msg = "Could not read refined peak widths"
                raise exceptions.PCRFileError(msg)
        # Search for unit-cell parameters
        cell_matches = self.cell_re.findall(summary)
        if len(cell_matches) < len(self.all_phases):
            raise exceptions.RefinementError()
        for idx, phase in enumerate(self.phases):
            match = cell_matches[idx].split()
            cell_params = [float(x) for x in match[::2]]
            # cell_stdevs = [float(x) for x in match[1::2]]
            unit_cell = phase.unit_cell
            unit_cell.a = cell_params[0]
            unit_cell.b = cell_params[1]
            unit_cell.c = cell_params[2]
            unit_cell.alpha = cell_params[3]
            unit_cell.beta = cell_params[4]
            unit_cell.gamma = cell_params[5]
    
    def pcrfile_context(self):
        """Generate a dict of values to put into a pcr input file."""
        context = {}
        num_phases = len(self.all_phases)
        context['num_phases'] = num_phases
        # Determine refinement mode based on number of phases
        if num_phases == 1:
            mode = Mode.constant_scale
        elif num_phases > 1:
            mode = Mode.constant_intensities
        context['refinement_mode'] = mode
        # Prepare parameters for each phase
        phases = []
        for phase in self.all_phases:
            unitcell = phase.unit_cell
            vals = {
                'a': unitcell.a, 'b': unitcell.b, 'c': unitcell.c,
                'alpha': unitcell.alpha, 'beta': unitcell.beta, 'gamma': unitcell.gamma,
                'u': phase.u, 'v': phase.v, 'w': phase.w, 'x': phase.x,
                'scale': phase.scale_factor, 'eta': phase.eta, 'I_g': phase.I_g,
                'Bov': phase.isotropic_temp,
            }
            # Codewords control which parameters are refined and in what order
            codewords = {
                'a': 0, 'b': 0, 'c': 0,
                'alpha': 0, 'beta': 0, 'gamma': 0,
                'u': 0, 'v': 0, 'w': 0, 'x': 0,
                'scale': 0, 'eta': 0
            }
            phases.append({
                'name': str(phase),
                'spacegroup': phase.fullprof_spacegroup,
                'vals': vals,
                'codewords': codewords
            })
        context['phases'] = phases
        # Background corrections
        context['bg_coeffs'] = self.bg_coeffs
        context['bg_codewords'] = [0 for x in self.bg_coeffs]
        # Instrument corrections
        context['zero'] = self.zero
        context['zero_codeword'] = 0
        context['displacement'] = self.displacement
        context['displacement_codeword'] = 0
        context['transparency'] = self.transparency
        context['transparency_codeword'] = 0
        # Meta-data
        context['num_params'] = 0
        return context
    
    def plot(self, ax=None):
        if ax is None:
            ax = plots.new_axes()
        df = self.calculated_diffractogram
        ax.plot(df[' 2Theta'], df['Yobs'])
        ax.plot(df[' 2Theta'], df['Ycal'])
        ax.plot(df[' 2Theta'], df['Yobs-Ycal'])
        ax.set_title('Profile refinement for {name}'.format(name=self.scan.axes_title()))
        ax.set_xlim(
            right=df[' 2Theta'].max()
        )
        return ax
    
    def details(self):
        msg = "Χ²: {chi}".format(chi=self.chi_squared)
        return msg
    
    def highlight_peaks(self, ax):
        """No-op for fullprof refinement."""
        return ax
    
    def broadenings(self, two_theta=None, intensities=None):
        """Retrieve the expected peak broadening for each phase.
        
        If refinement of the scale_factor has not been done, this
        method will first perform the refinement process. Currently,
        this improperly uses instrument broadening, so the value for
        each phase is the same. The specific value is the anticipated
        FWHM at the median angle in ``two_theta``.
        
        Parameters
        ----------
        two_theta : np.ndarray, optional
          An array of 2θ° values.
        intensities : np.ndarray, optional
          Observed diffraction intensities for the 2θ values. Used for
          refinement if necessary.
        
        Returns
        -------
        broadenings : tuple
          The predicted peak broadenings for each phase.
        
        """
        if not self.is_refined['peak_widths']:
            self.refine_all(two_theta, intensities)
        angle = np.median(two_theta)
        angle = math.radians(angle)
        broadenings = []
        for phase in self.phases:
            # Calculate FHWM for each phase
            width_squared = phase.u * math.tan(angle)**2 + phase.v * math.tan(angle)**2 + phase.w
            if width_squared >= 0:
                width = math.sqrt(width_squared)
            else:
                width = 0
            broadenings.append(width)
        return broadenings
    
    def predict(self, two_theta):
        """Return predicted diffraction intensities for given 2θ.
        
        The pattern created by refinement will the interpolated to the
        new ``two_theta`` values. For this reason, it is important
        that ``two_theta`` is similar to the range used for running
        the refinements.
        
        Parameters
        ----------
        two_theta : np.ndarray
          Diffraction angles (2θ°) for predicting the diffraction
          pattern.
        
        Returns
        -------
        predicted : np.ndarray
          Diffraction intensities for values in ``two_theta``.
        
        """
        calc = self.calculated_diffractogram
        predicted = self.respline(x=calc[' 2Theta'], y=calc['Ycal'],
                                  new_x=two_theta)
        return predicted
    
    def goodness_of_fit(self, two_theta=None, intensities=None):
        """Retrieve the degree of goodness for the refinement.
        
        The meaning of "goodness" depends on the specific type of
        refinement.
        
        Parameters
        ----------
        two_theta : np.ndarray, optional
          An array of 2θ° values.
        intensities : np.ndarray, optional
          Observed diffraction intensities for the 2θ values. Used for
          refinement if necessary.
        
        Returns
        -------
        goodness : np.ndarray
          A value describing how reliable the fit it.
        
        """
        if self.chi_squared is not None:
            confidence = self.chi_squared
        else:
            confidence = np.inf
        return confidence
