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

    def plot_cycle(self, xcolumn, ycolumn, ax, label=None):
        # Default label for plot
        if label is None:
            label = "Cycle {}".format(self.number)
        # Drop missing data
        df = self.df.dropna(subset=[xcolumn, ycolumn])
        # Plot remaining values
        ax.plot(df[xcolumn], df[ycolumn], label=label)
        return ax
