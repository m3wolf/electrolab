# -*- coding: utf-8 -*-

import pandas as pd

import plots

class FullProfProfileRefinement():

    def __init__(self, filename):
        self.filename = filename

    def calculated_diffractogram(self):
        """Read a pcf file and return the refinement as a dataframe."""
        df = pd.read_csv(self.filename, skiprows=3, sep='\t')
        return df

    def plot(self, ax=None):
        if ax is None:
            ax = plots.new_axes()
        df = self.calculated_diffractogram()
        ax.plot(df[' 2Theta'], df['Yobs'])
        ax.plot(df[' 2Theta'], df['Ycal'])
        ax.plot(df[' 2Theta'], df['Yobs-Ycal'])
        ax.set_title('Profile refinement {filename}'.format(filename=self.filename))
        ax.set_xlim(
            right=df[' 2Theta'].max()
        )
        return ax
