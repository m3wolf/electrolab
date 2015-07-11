# -*- coding: utf-8 -*-

from collections import namedtuple
import math

import numpy as np
from scipy import optimize
from matplotlib import pyplot

import exceptions

# kalpha2 is half the intensity of kalpha1
KALPHA2_RATIO = 0.5

class XrdTube():
    def __init__(self, kalpha1, kalpha2):
        self.kalpha1 = kalpha1
        self.kalpha2 = kalpha2

    @property
    def kalpha(self):
        wavelength = (self.kalpha1 + KALPHA2_RATIO*self.kalpha2)/(1+KALPHA2_RATIO)
        return wavelength

    def split_angle_by_kalpha(self, angle):
        """Predict kα1/kα2 splitting at the given 2θ angle."""
        theta1 = math.degrees(
            math.asin(self.kalpha1*math.sin(math.radians(angle))/self.kalpha)
        )
        theta2 = math.degrees(
            math.asin(self.kalpha2*math.sin(math.radians(angle))/self.kalpha)
        )
        return (theta1, theta2)

tubes = {
    'Cu': XrdTube(kalpha1=1.5406, kalpha2=1.5444),
}

class PeakFit():
    Parameters = namedtuple('Parameters', ('height', 'center', 'width'))
    height = 450
    center = 35.15
    width = 0.02

    def __repr__(self):
        return "<{cls}: {two_theta}°>".format(cls=self.__class__.__name__,
                                              two_theta=round(self.center, 2))

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
        return self.kernel(x, **self.parameters.__dict__)

    def initial_parameters(self, xdata, ydata, tube=tubes['Cu']):
        # Determine center of mass for the peak (plus correction for lower k-alpha2 weights)
        mean_center = np.average(xdata, weights=ydata)
        # Determine centers for k-alpha1 and k-alpha2
        center1, center2 = tube.split_angle_by_kalpha(mean_center)
        # Determine maximum peak height
        maxHeight = ydata.max()
        # Determine full-width half-max
        stdDev = self.width
        # Prepare tuples of parameters
        p1 = self.Parameters(height=maxHeight,
                             center=center1,
                             width=stdDev)
        p2 = self.Parameters(height=maxHeight/2,
                             center=center2,
                             width=stdDev)
        return (p1, p2)


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
    width_g = 0.2
    width_c = 0.2
    eta = 0.5
    Parameters = namedtuple('PseudoVoigtParameters', ('height_g', 'height_c',
                                                      'center',
                                                      'width_g', 'width_c',
                                                      'eta'))

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

    def initial_parameters(self, xdata, ydata, tube=tubes['Cu']):
        # Determine center of mass for the peak (plus correction for lower k-alpha2 weights)
        mean_center = np.average(xdata, weights=ydata)
        # Determine centers for k-alpha1 and k-alpha2
        center1, center2 = tube.split_angle_by_kalpha(mean_center)
        # Determine maximum peak height
        maxHeight = ydata.max()
        # Determine full-width half-max
        stdDev = self.width
        # Prepare tuples of parameters
        p1 = self.Parameters(height_g=maxHeight,
                             height_c=maxHeight,
                             center=center1,
                             width_g=stdDev,
                             width_c=stdDev,
                             eta=0.5)
        p2 = self.Parameters(height_g=maxHeight/2,
                             height_c=maxHeight/2,
                             center=center2,
                             width_g=stdDev,
                             width_c=stdDev,
                             eta=0.5)
        return (p1, p2)

    @staticmethod
    def kernel(x, height_g, height_c, center, width_g, width_c, eta):
        """
        Compute a linear combination of Gaussian and Cachy functions:
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
    A peak in an X-ray diffractogram. May be composed of multiple
    overlapping subpeaks from different wavelengths.
    """
    fit_list = []

    def __init__(self, reflection=None):
        self.reflection=reflection

    def __repr__(self):
        name = "<{cls}: {angle}°>".format(
            cls=self.__class__.__name__,
            angle=self.center_mean
        )
        return name

    @property
    def center_mean(self):
        """Determine the average peak position based on fits."""
        if len(self.fit_list) > 0:
            total = sum([fit.center for fit in self.fit_list])
            center = total/len(self.fit_list)
        else:
            center = None
        return center

    @property
    def center_kalpha(self):
       """Determine the peak center based on relative intensities of kalpha1
       and kalpha2."""
       total = self.fit_list[0].center + KALPHA2_RATIO*self.fit_list[1].center
       center = total/(1+KALPHA2_RATIO)
       return center

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

    def fit(self, two_theta, intensity, num_peaks=2, method='pseudo-voigt'):
        """Least squares refinement of a function to the data in two_theta
        and intensity. Method can be any of the following peak shapes:
            - 'Gaussian'
            - 'Cauchy'
            - 'Pearson VII'
            - 'Pseudo-Voigt'
        """
        fitClasses = {
            'gaussian': GaussianFit,
            'cauchy': CauchyFit,
            'pearson vii': PearsonVIIFit,
            'pseudo-voigt': PseudoVoigtFit,
            'estimated': EstimatedFit,
        }
        # Save two_theta range for later
        self.two_theta_range = (two_theta[0], two_theta[-1])
        # Create fit object(s)
        self.fit_list = []
        FitClass = fitClasses[method.lower()]
        for i in range(0, num_peaks):
            self.fit_list.append(FitClass())
        # Define objective function
        def objective(two_theta, *params):
            # Unpack the parameters
            paramGroups = self.split_parameters(params)
            result = np.zeros_like(two_theta)
            for idx, fit in enumerate(self.fit_list):
                y = fit.kernel(two_theta, *paramGroups[idx])
                result += y
            return result
        # Compute initial parameters
        initialParameters = FitClass().initial_parameters(xdata=two_theta,
                                                          ydata=intensity)
        initialParameters = initialParameters[0] + initialParameters[1]
        # Minimize the residual least squares
        try:
            popt, pcov = optimize.curve_fit(objective,
                                            xdata=two_theta,
                                            ydata=intensity,
                                            p0=initialParameters)
        except RuntimeError as e:
            # Could not find optimum fit
            angle = (self.two_theta_range[0]+self.two_theta_range[1])/2
            msg = "Peak ~{angle:.1f}°: {error}".format(angle=angle, error=e)
            raise exceptions.PeakFitError(msg)
        else:
            # Split optimized parameters by number of fits
            paramsList = self.split_parameters(popt)
            # Save optimized parameters for each fit
            for idx, fit in enumerate(self.fit_list):
                fit.parameters = paramsList[idx]
            return pcov

    def x_range(self):
        """Return a range of x values over which this fit is reasonably defined."""
        x = np.linspace(self.two_theta_range[0],
                        self.two_theta_range[1],
                        num=1000)
        return x

    def plot_overall_fit(self, ax=None, background=None):
        x = self.x_range()
        if ax is None:
            ax = pyplot.gca()
        y = np.zeros_like(x)
        for fit in self.fit_list:
            y += fit.evaluate(x)
            if background is not None:
                y += background(x)
        ax.plot(x, y, label="overall fit")

    def plot_fit(self, ax=None, background=None):
        """Plot the subpeaks on the given axes. background(x) will be added to
        each peak."""
        x = self.x_range()
        if ax is None:
            ax = pyplot.gca()
        for fit in self.fit_list:
            y = fit.evaluate(x)
            if background is not None:
                y = y+background(x)
            ax.plot(x, y, label="fit subpeak")
