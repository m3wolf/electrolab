# -*- coding: utf-8 -*-
#
# Copyright © 2016 Mark Wolf
#
# This file is part of scimap.
#
# Scimap is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# Scimap is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with Foobar.  If not, see <http://www.gnu.org/licenses/>.

"""Routines related to fitting curves to data (eg fitting XANES data
with a Gaussian curve."""

from collections import namedtuple
import math

import numpy as np
from scipy import optimize
import pandas as pd

import plots

# How strongly to penalize negative peak heights, etc
BASE_PENALTY = 300

def remove_peak_from_df(reflection, df):
    """Accept an xrd scan dataframe and remove the given reflection's peak from
    the data."""
    peak = reflection.two_theta_range
    df.drop(df[peak[0]:peak[1]].index, inplace=True)

def gaussian_fwhm(width_parameter):
    """Calculates full-width half max based on width parameter."""
    # Taken from wikipedia page for "Gaussian Function".
    return 2.35482 * width_parameter

def cauchy_fwhm(width_parameter):
    """Calculates full-width half max based on width parameter."""
    return 2*width_parameter

class PeakFit():
    Parameters = namedtuple('Parameters', ('height', 'center', 'width'))
    height = 450
    center = 35.15
    width = 0.02

    def __repr__(self):
        return "<{cls}: {center}>".format(cls=self.__class__.__name__,
                                              center=round(self.center, 2))

    @property
    def parameters(self):
        return self.Parameters(self.height, self.center, self.width)

    @parameters.setter
    def parameters(self, value):
        params = self.Parameters(*value)
        self.height = params.height
        self.center = params.center
        self.width = params.width

    def evaluate(self, x):
        """Evaluate this fitted subpeak at given x values."""
        return self.kernel(x, **self.parameters._asdict())

    def penalty(self, params):
        """Rules for contraining the fitting algorithm. 0 means no penalty."""
        penalty = 0
        # Penalize negative peak heights
        if params.height < 0:
                penalty += BASE_PENALTY
        if params.width < 0:
                penalty += BASE_PENALTY
        return penalty

    def initial_parameters(self, xdata, ydata, center=0):
        """Estimate intial parameters from data. Calling function is
        responsible for determining peak center since multiple peaks
        may be involved.
        """
        # Determine maximum peak height
        maxHeight = ydata.max()

        # Determine full-width half-max
        # Split the dataset into an upper half and a lower half
        maxidx = ydata.argmax()
        rightdata = ydata[maxidx:]
        leftdata = ydata[:maxidx]
        # Find the nearest datum to halfmax in each half
        halfmax = ydata.max() / 2
        rightidx = (np.abs(rightdata - halfmax)).argmin()
        leftidx = (np.abs(leftdata - halfmax)).argmin()
        # Subtract the two points to get full-width-half-max
        rightx = xdata[rightidx + maxidx]
        leftx = xdata[leftidx]

        # Convert FWHM to stdDev (taken from wolfram alpha page for Gaussian)
        stdDev = (leftx - rightx) / 2.3548

        # Prepare tuples of parameters
        p1 = self.Parameters(height=maxHeight,
                             center=center,
                             width=stdDev)
        return p1

class EstimatedFit(PeakFit):
    """Fallback fit using just estimated intial parameters."""
    pass

class GaussianFit(PeakFit):
    @staticmethod
    def kernel(x, height, center, width):
        """
        Compute a Gaussian distribution of peak height and width around center.
        x is an array of points for which to return y values.
        """
        y = height * np.exp(-np.square(x-center)/2/np.square(width))
        return y


class CauchyFit(PeakFit):

    @staticmethod
    def kernel(x, height, center, width):
        """
        Compute a Cauchy (Lorentz) distribution of peak height and width
        around center.  x is an array of points for which to return y
        values.
        """
        y = height * np.square(width)/(np.square(width)+np.square(x-center))
        return y


class PearsonVIIFit(PeakFit):
    @staticmethod
    def kernel(x, height, center, width, exponent):
        raise NotImplementedError


class PseudoVoigtFit(PeakFit):
    height_g = 450
    height_c = 450
    center = 35.15
    width_g = 0.05
    width_c = 0.05
    eta = 0.5
    Parameters = namedtuple('PseudoVoigtParameters', ('height_g', 'height_c',
                                                      'center',
                                                      'width_g', 'width_c',
                                                      'eta'))

    @property
    def height(self):
        return self.height_g + self.height_c

    @property
    def width(self):
        return self.width_g + self.width_c

    def fwhm(self):
        # Gaussian component
        fwhm = self.eta * gaussian_fwhm(self.width_g)
        # Cauchy component
        fwhm += (1-self.eta) * cauchy_fwhm(self.width_c)
        return fwhm

    @property
    def parameters(self):
        return self.Parameters(height_g=self.height_g,
                               height_c=self.height_c,
                               center=self.center,
                               width_g=self.width_g,
                               width_c=self.width_c,
                               eta=self.eta)

    @parameters.setter
    def parameters(self, value):
        params = self.Parameters(*value)
        self.height_g = params.height_g
        self.height_c = params.height_c
        self.center = params.center
        self.width_g = params.width_g
        self.width_c = params.width_c
        self.eta = params.eta

    def penalty(self, params):
        penalty = 0
        # Prepare parameters for penalty from parent class
        parent = super(PseudoVoigtFit, self)
        gParams = parent.Parameters(height=params.height_g,
                                    center=params.center,
                                    width=params.width_g)
        penalty += parent.penalty(gParams)
        cParams = parent.Parameters(height=params.height_c,
                                    center=params.center,
                                    width=params.width_c)
        penalty += parent.penalty(cParams)
        # Check for eta between zero and 1
        if not( 0 < params.eta < 1):
            penalty += BASE_PENALTY
        return penalty

    def initial_parameters(self, xdata, ydata, center=0):
        """Estimate intial parameters from data. Calling function is
        responsible for determining peak center since multiple peaks
        may be involved.
        """
        # Determine maximum peak height
        maxHeight = ydata.max()
        # Determine full-width half-max
        print("TODO: Estimate initial peak width")
        stdDev = self.width
        # Prepare tuples of parameters
        params = self.Parameters(height_g=maxHeight,
                                 height_c=maxHeight,
                                 center=center,
                                 width_g=stdDev,
                                 width_c=stdDev,
                                 eta=0.5)
        return params


    @staticmethod
    def kernel(x, height_g, height_c, center, width_g, width_c, eta):
        """
        Compute a linear combination of (G)aussian and (C)achy functions:
            y = eta*G + (1-eta)*C
        params are tuples of (height, center, width) to pass to the respective
        functions. x is an array of points for which to return y
        values.
        """
        g = GaussianFit.kernel(x, height_g, center, width_g)
        c = CauchyFit.kernel(x, height_c, center, width_c)
        y = eta*g + (1-eta)*c
        return y


class Peak():
    """
    A single peak in data. The actual peak types (Gaussian, Cauchy,
    etc.) are described in PeakFit objects.
    """
    vertical_offset = 0
    def split_parameters(self, params):
        """
        Take a full list of parameters and divide it groups for each subpeak.
        """
        numFits = len(self.fit_list)
        chunkSize = int(len(params)/numFits)
        groups = []
        for i in range(0, len(params), chunkSize):
            groups.append(params[i:i+chunkSize])
        return groups

    def fit(self, x, y, num_peaks=1, method='pseudo-voigt'):
        """Least squares refinement of a function to the data.

        Arguments
        ---------
        - x (array-like) : Array of data to be fit along the
          independent variable.

        - y (array-like) : Array of data to be fit along the dependent
          variable.

        - num_peaks (int) : How many overlapping peaks should be
          used. Eg. X-ray data often has kα1 and kα2 peaks (default 1)

        - method (str) : Selects which peak shape to use. Valid choices are:
            - 'Gaussian'
            - 'Cauchy'
            - 'Pearson VII'
            - 'Pseudo-Voigt'
            - 'estimated'
        """
        fitClasses = {
            'gaussian': GaussianFit,
            'cauchy': CauchyFit,
            'pearson vii': PearsonVIIFit,
            'pseudo-voigt': PseudoVoigtFit,
            'estimated': EstimatedFit,
        }
        # Have not yet determined correct way to handle multiple peaks
        if not num_peaks == 1:
            raise NotImplementedError('Fitting multiple peaks not yet ready.')
        # Save x range for later
        self.x_range = (x[0], x[-1])
        # Create fit object(s)
        self.fit_list = []
        FitClass = fitClasses[method.lower()]
        for i in range(0, num_peaks):
            self.fit_list.append(FitClass())
        # Define objective function
        def objective(x, *params):
            # Unpack the parameters
            paramGroups = self.split_parameters(params)
            result = np.zeros_like(x)
            for idx, fit in enumerate(self.fit_list):
                y = fit.kernel(x, *paramGroups[idx])
                result += y
            return result
        # Error function, penalizes values out of range
        def residual_error(obj_params):
            penalty = 0
            # Calculate dual peak penalties
            paramlist = self.split_parameters(obj_params)
            paramlist = [FitClass.Parameters(*p) for p in paramlist]
            for fit, paramTuple in zip(self.fit_list, paramlist):
                # Calculate single peak penalties
                penalty += fit.penalty(paramTuple)
            result = objective(x, *obj_params)
            return (y-result)**2+penalty
        # Guess reasonable values for initial parameters
        max_idx = y.argmax()
        peak_max = x[max_idx]
        initialParameters = FitClass().initial_parameters(xdata=x,
                                                          ydata=y,
                                                          center=peak_max)
        # Minimize the residual least squares
        try:
            result = optimize.leastsq(residual_error, x0=initialParameters,
                                      full_output=True)
        except RuntimeError as e:
            # Could not find optimum fit
            angle = (self.x_range[0]+self.x_range[1])/2
            msg = "Peak ~{angle:.1f}°: {error}".format(angle=angle, error=e)
            raise exceptions.PeakFitError(msg)
        else:
            popt = result[0]
            # self.residuals = result[2]['fvec'].sum()
            # Split optimized parameters by number of fits
            paramsList = self.split_parameters(popt)
            # Save optimized parameters for each fit
            for idx, fit in enumerate(self.fit_list):
                fit.parameters = paramsList[idx]
            # return self.residuals


    def predict(self, xdata=None):
        """Get a dataframe of the predicted peak fits.

        Arguments
        ---------
        xdata (numpy array) : An array of x values to use as the index
            for predicting y values. If None (default), a numpy linspace
            will be generated within the range intially used for the
            fitting.
        """
        if xdata is None:
            # Create a default linspace to use as xdata
            x = np.linspace(self.x_range[0],
                            self.x_range[1],
                            num=1000)
        else:
            x = xdata
        y = np.zeros_like(x)
        for fit in self.fit_list:
            y += fit.evaluate(x)
            # if background is not None:
            #     # I don't know why I have to divide by 2 here but it works(!?)
            #     y += background(x)/2
        # Correct for vertical offset
        y += self.vertical_offset
        return pd.Series(data=y, index=x)

    def residuals(self, observations):
        """Calculate the differences at each point between the fit and the
        provided data.

        Arguments
        ---------
        observations (pandas series): The original data against which
            to compare the fit.
        """
        predicted = self.predict(xdata=observations.index)
        residuals = observations - predicted
        return residuals

    def goodness(self, observations):
        """Calculate the goodness of fit. Returns the sum of squared residuals
        divided by degrees of freedom. Lower numbers describe better
        fit.

        Arguments
        ---------
        observations (pandas series): The original data against which
            to compare the fit.
        """
        predicted = self.predict(xdata=observations.index)
        print(predicted)
        print(observations)
        # Determine total residual
        sum_of_squares = (self.residuals(observations)**2).sum()
        # Divide by degrees of freedom
        goodness = math.sqrt(sum_of_squares) / (len(observations)-1)
        return goodness

    def center(self):
        centers = [f.center for f in self.fit_list]
        mean_center = sum(centers) / len(centers)
        return mean_center

    def plot_fit(self, ax=None, background=None):
        df = self.predict()
        if ax is None:
            ax = plots.new_axes()
        ax.plot(df.index, df.values, label="overall fit")
        return ax
