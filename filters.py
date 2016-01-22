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

"""Collection of functions for transforming data that are not
specific to any one technique.
"""

import numpy as np
import pandas as pd
from scipy import fftpack

def fourier_transform(data):
    """Perform a discrete fourier transform on the data. Input should be a
    pandas Series(). Assumes data are all real. Returns a new series
    with the data in the frequency domain.
    """
    # Transform y data
    yf = fftpack.rfft(data.values)
    # Transform x data
    x = data.index
    T = (x.max()-x.min())/len(x) # Spacing of data
    N = len(data) # Number of data
    xf = np.linspace(0, 1/(2*T), N)
    # Return new data series
    newData = pd.Series(data=yf, index=xf)
    return newData

def inverse_fourier_transform(data):
    """Perform an inverse discrete fourier transform on the data. Input should
    be a pandas Series(). Assumes data are all real. Returns a Series."""
    # Transform y data
    yf = fftpack.irfft(data.values)
    # Transform x data
    x = data.index
    T = (x.max()-x.min())/len(x) # Spacing of data
    N = len(data) # Number of data
    xf = np.linspace(0, 1/(2*T), N)
    # Return new data series
    newData = pd.Series(data=yf, index=xf)
    return newData


class Filter():
    def __init__(self, cutoff=10):
        self.cutoff=cutoff

    def difference(self, data):
        """
        Return the difference between the original data and filtered data.
        """
        filtered = self.apply(data)
        difference = data - filtered
        return difference


class LowPassFilter(Filter):
    """Applies a sharp cutoff filter and allows low frequencies to pass
    unmodified and high frequencies set to 0.
    """
    def apply(self, data):
        # Transform data into frequency space
        fData = fourier_transform(data)
        # Build the transfer function
        transferFunction = fData.copy()
        for idx in transferFunction.index:
            if idx < self.cutoff:
                transferFunction[idx] = 1
            else:
                transferFunction[idx] = 0
        # Apply filter
        filtered = fData*transferFunction
        # Transform back to 2-theta space
        filtered = pd.Series(inverse_fourier_transform(filtered).values, data.index)
        return filtered

class HighPassFilter(Filter):
    """Applies a sharp cutoff filter and allows high frequencies to pass
    unmodified and low frequencies set to 0.
    """
    def apply(self, data):
        # Transform data into frequency space
        fData = fourier_transform(data)
        # Build the transfer function
        transferFunction = fData.copy()
        for idx in transferFunction.index:
            if idx > self.cutoff:
                transferFunction[idx] = 1
            else:
                transferFunction[idx] = 0
        # Apply filter
        filtered = fData*transferFunction
        # Transform back to 2-theta space
        filtered = pd.Series(inverse_fourier_transform(filtered).values, data.index)
        return filtered
