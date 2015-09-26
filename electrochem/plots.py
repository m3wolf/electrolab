import units

from scimap.plots import new_axes
from . import electrochem_units

def plot_rate_capacities(runs, ax=None, cycle_idx=0):
    """Plot the discharge capacity as a function of charge-rate for a set
    of runs. Similar in spirit to a ragone plot."""
    if ax is None:
        ax = new_axes()
    # Arrays for holding data to be plotted
    capacities = []
    rates = []
    # Fill arrays with data
    for run in runs:
        # Determine capacity
        capacity = run.charge_capacity(cycle_idx=cycle_idx)
        capacities.append(capacity)
        # Determine C-rate (with units)
        time = run.theoretical_capacity / abs(run.charge_current)
        rate = 1/(electrochem_units.hour(time))
        rates.append(rate)
    # Plot resulting data
    ax.plot(rates, capacities, marker='o')
    ax.set_xscale('log')
    ax.set_ylim(bottom=0)
    ax.set_xlabel('C-rate / $h^{-1}$')
    ax.set_ylabel('Charge capacity / $mAhg^{-1}$')
    ax.set_title('Rate-capacities during cycle {}'.format(cycle_idx + 1))
    return ax
