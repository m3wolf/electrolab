# -*- coding: utf-8 -*-

from collections import namedtuple

import pandas
import numpy as np
from scipy import optimize
from matplotlib import pyplot

import exceptions
from xrd.tube import tubes, KALPHA2_RATIO
import plots


# class XRDPeak():
#     """
#     A peak in an X-ray diffractogram. May be composed of multiple
#     overlapping subpeaks from different wavelengths.
#     """
#     fit_list = []

#     def __init__(self, reflection=None):
#         self.reflection = reflection

#     def __repr__(self):
#         name = "<{cls}: {angle}°>".format(
#             cls=self.__class__.__name__,
#             angle=self.center_mean
#         )
#         return name

#     @property
#     def center_mean(self):
#         """Determine the average peak position based on fits."""
#         if len(self.fit_list) > 0:
#             total = sum([fit.center for fit in self.fit_list])
#             center = total/len(self.fit_list)
#         else:
#             center = None
#         return center

#     @property
#     def center_kalpha(self):
#         """Determine the peak center based on relative intensities of kalpha1
#         and kalpha2."""
#         total = self.fit_list[0].center + KALPHA2_RATIO*self.fit_list[1].center
#         center = total/(1+KALPHA2_RATIO)
#         return center

#     def fwhm(self):
#         """Full width at half-maximum. Currently, uses the sum of kα1 and kα2,
#         which is not accurate if the peaks overlap significantly.
#         """
#         width = 0
#         for fit in self.fit_list:
#             width += fit.fwhm()
#         return width

#     def split_parameters(self, params):
#         """
#         Take a full list of parameters and divide it groups for each subpeak.
#         """
#         numFits = len(self.fit_list)
#         chunkSize = int(len(params)/numFits)
#         groups = []
#         for i in range(0, len(params), chunkSize):
#             groups.append(params[i:i+chunkSize])
#         return groups

#     def fit(self, two_theta, intensity, num_peaks=2, method='pseudo-voigt'):
#         """Least squares refinement of a function to the data in two_theta
#         and intensity. Method can be any of the following peak shapes:
#             - 'Gaussian'
#             - 'Cauchy'
#             - 'Pearson VII'
#             - 'Pseudo-Voigt'
#         """
#         fitClasses = {
#             'gaussian': GaussianFit,
#             'cauchy': CauchyFit,
#             'pearson vii': PearsonVIIFit,
#             'pseudo-voigt': PseudoVoigtFit,
#             'estimated': EstimatedFit,
#         }
#         # Save two_theta range for later
#         self.two_theta_range = (two_theta[0], two_theta[-1])
#         # Create fit object(s)
#         self.fit_list = []
#         FitClass = fitClasses[method.lower()]
#         for i in range(0, num_peaks):
#             self.fit_list.append(FitClass())
#         # Define objective function
#         def objective(two_theta, *params):
#             # Unpack the parameters
#             paramGroups = self.split_parameters(params)
#             result = np.zeros_like(two_theta)
#             for idx, fit in enumerate(self.fit_list):
#                 y = fit.kernel(two_theta, *paramGroups[idx])
#                 result += y
#             return result
#         # Error function, penalizes values out of range
#         def residual_error(obj_params):
#             penalty = 0
#             # Calculate dual peak penalties
#             params1, params2 = self.split_parameters(obj_params)
#             params1 = FitClass.Parameters(*params1)
#             params2 = FitClass.Parameters(*params2)
#             # if not (params1.height*0.4 < params2.height < params1.height*0.6):
#             #     penalty += BASE_PENALTY
#             for fit, paramTuple in zip(self.fit_list, [params1, params2]):
#                 # Calculate single peak penalties
#                 penalty += fit.penalty(paramTuple)
#             result = objective(two_theta, *obj_params)
#             return (intensity-result)**2+penalty
#         # Compute initial parameters
#         initialParameters = FitClass().initial_parameters(xdata=two_theta,
#                                                           ydata=intensity)
#         initialParameters = initialParameters[0] + initialParameters[1]
#         # Minimize the residual least squares
#         try:
#             # popt, pcov = optimize.curve_fit(objective,
#             #                                 xdata=two_theta,
#             #                                 ydata=intensity,
#             #                                 p0=initialParameters)
#             result = optimize.leastsq(residual_error, x0=initialParameters,
#                                       full_output=True)
#         except RuntimeError as e:
#             # Could not find optimum fit
#             angle = (self.two_theta_range[0]+self.two_theta_range[1])/2
#             msg = "Peak ~{angle:.1f}°: {error}".format(angle=angle, error=e)
#             raise exceptions.PeakFitError(msg)
#         else:
#             popt = result[0]
#             residual = result[2]['fvec'].sum()
#             # Split optimized parameters by number of fits
#             paramsList = self.split_parameters(popt)
#             # Save optimized parameters for each fit
#             for idx, fit in enumerate(self.fit_list):
#                 fit.parameters = paramsList[idx]
#             return residual

#     def x_range(self):
#         """Return a range of x values over which this fit is reasonably defined."""
#         x = np.linspace(self.two_theta_range[0],
#                         self.two_theta_range[1],
#                         num=1000)
#         return x

#     def dataframe(self, background=None):
#         """Get a dataframe of the predicted peak fits."""
#         x = self.x_range()
#         y = np.zeros_like(x)
#         for fit in self.fit_list:
#             y += fit.evaluate(x)
#             if background is not None:
#                 # I don't know why I have to divide by 2 here but it works(!?)
#                 y += background(x)/2
#         return pandas.DataFrame(data=y, index=x, columns=['counts'])

#     def plot_overall_fit(self, ax=None, background=None):
#         df = self.dataframe(background=background)
#         if ax is None:
#             ax = plots.new_axes()
#         ax.plot(df.index, df.counts, label="overall fit")

#     def plot_fit(self, ax=None, background=None):
#         """Plot the subpeaks on the given axes. background(x) will be added to
#         each peak."""
#         x = self.x_range()
#         if ax is None:
#             ax = pyplot.gca()
#         for fit in self.fit_list:
#             y = fit.evaluate(x)
#             if background is not None:
#                 y = y+background(x)
#             ax.plot(x, y, label="fit subpeak")
