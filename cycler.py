from matplotlib import pyplot
import numpy as np
import pandas as pd

def axis_label(key):
    axis_labels = {
        'Ewe/V': r'$E\ /V$',
        'capacity': r'$Capacity\ / mAhg^{-1}$',
    }
    return axis_labels.get(key, key)

def new_axes():
    """Create a new set of matplotlib axes for plotting"""
    fig = pyplot.figure(figsize=(10, 6))
    ax = pyplot.gca()
    return ax

class GalvanostatRun():
    """
    Electrochemical experiment cycling on one channel.
    Galvanostatic control potential limited (GPLC).
    """
    cycles = []

    def __init__(self, df, mass=None, *args, **kwargs):
        self._df = df
        all_cycles = self._df.loc[self._df['mode']!=3]
        # Calculate capacity from charge
        if mass:
            all_cycles.loc[:,'capacity'] = all_cycles.loc[:,'(Q-Qo)/mA.h']/mass
        # Split the data into cycles, except the initial resting phase
        if 'cycle number' in all_cycles.columns:
            self.cycles = list(all_cycles.groupby('cycle number'))
        else:
            self.cycles = [(0, all_cycles)]
        super(GalvanostatRun, self).__init__(*args, **kwargs)

    def plot_cycles(self, xcolumn, ycolumn, ax=None):
        """Plot each electrochemical cycle"""
        if not ax:
            ax = new_axes()
        ax.set_xlabel(axis_label(xcolumn))
        ax.set_ylabel(axis_label(ycolumn))
        for cycle in self.cycles:
            ax.plot(cycle[1][xcolumn], cycle[1][ycolumn])
        length = len(self.cycles)
        ax.legend(range(1, length))
        return ax

    def plot_dq_dE(self, ax=None):
        """Plot the first derivative of each of the cycles"""
        # Prepare defaults
        if not ax:
            ax = new_axes()
        for cycle in self.cycles:
            dE = cycle[1].loc[:, 'Ewe/V']
            q = cycle[1].loc[:, 'capacity']*1000
            # q = cycle[1].loc[:, 'dq/mA.h']*1000
            dq_dE = np.gradient(q, dE)
            df = pd.DataFrame({'dE': dE, 'dq_dE': dq_dE})
            # df = df[df.dq_dE<0]
            ax.plot(df['dE'], df['dq_dE'])
        ax.set_xlabel(axis_labels['Ewe/V'])
        ax.set_ylabel(r'$\frac{dQ}{dE}\ / Ahg^{-1}V^{-1}$')
        length = len(self.cycles)
        ax.legend(range(1, length))
        return ax

class Cycle():
    """Data from one charge-discharge cycle."""
    pass
