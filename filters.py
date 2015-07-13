# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
from scipy import fftpack

def fourier_transform(data):
    """Perform a discrete fourier transform on the data. Input should
    be a pandas Series(). Assumes data are all real."""
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
    be a pandas Series(). Assumes data are all real. Returns a Series"""
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
    unmodified and high frequencies set to 0."""
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
