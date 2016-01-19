# -*- coding: utf-8 -*-

import numpy as np

import default_units

class Cycle():
    """Data from one charge-discharge cycle."""
    def __init__(self, number, df):
        self.number = number
        self.df = df

    def charge_capacity(self):
        """Calculate difference between discharged and charged state"""
        max_capacity = np.max(self.df['capacity'])
        min_idx = self.df['capacity'].first_valid_index()
        min_capacity = self.df['capacity'][min_idx]
        return max_capacity - min_capacity

    def discharge_capacity(self):
        """Calculate the difference between charged and discharged state"""
        max_capacity = np.max(self.df['capacity'])
        min_idx = self.df['capacity'].last_valid_index()
        min_capacity = self.df['capacity'][min_idx]
        return max_capacity - min_capacity

    def plot_cycle(self, xcolumn, ycolumn, ax, label=None, *args, **kwargs):
        # Default label for plot
        if label is None:
            label = "Cycle {}".format(self.number)
        # Check that the columns given exist in the dataframe
        error_msg = "Cannot find {dim}column. Choices are {choices}"
        if xcolumn not in self.df.columns:
            msg = error_msg.format(dim='x', choices=self.df.columns)
            raise KeyError(msg)
        if ycolumn not in self.df.columns:
            msg = error_msg.format(dim='y', choices=self.df.columns)
            raise KeyError(msg)
        # Drop missing data
        df = self.df.dropna(subset=[xcolumn, ycolumn])
        # Plot remaining values
        ax.plot(df[xcolumn], df[ycolumn], label=label, *args, **kwargs)
        return ax
