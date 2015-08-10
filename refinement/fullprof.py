# -*- coding: utf-8 -*-

import pandas as pd

import plots

class ProfileMatch():

    def __init__(self, scan):
        self.scan = scan
        # self.filename = filename

    def calculated_diffractogram(self):
        """Read a pcf file and return the refinement as a dataframe."""
        df = pd.read_csv(self.filename, skiprows=3, sep='\t')
        return df

    def pcrfile_context(self):
        """Generate a dict of values to put into a pcr input file."""
        context = {}
        phases = []
        for phase in self.scan.phases:
            unitcell = phase.unit_cell
            values = {
                'a': unitcell.a, 'b': unitcell.b, 'c': unitcell.c,
                'alpha': unitcell.alpha, 'beta': unitcell.beta, 'gamma': unitcell.gamma,
                'u': phase.u, 'v': phase.v, 'w': phase.w,
                'scale': phase.scale_factor, 'eta': phase.eta
            }
            # Codewords control which parameters are refined and in what order 
            codewords = {
                'a': 0, 'b': 0, 'c': 0,
                'alpha': 0, 'beta': 0, 'gamma': 0,
                'u': 0, 'v': 0, 'w': 0,
                'scale': 0, 'eta': 0
            }
            phases.append({
                'name': 'hello',
                'spacegroup': phase.fullprof_spacegroup,
                'values': values,
                'codewords': codewords
            })
        context['phases'] = phases
        return context

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
