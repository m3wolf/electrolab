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
# along with Scimap.  If not, see <http://www.gnu.org/licenses/>.

import re
import os

import units
import units.predefined
import pytz

import exceptions
from electrochem.cycle import Cycle
from plots import new_axes
import default_units
from . import biologic


def axis_label(key):
    axis_labels = {
        'Ewe/V': r'$E\ /V$',
        'capacity': r'$Capacity\ / mAhg^{-1}$',
    }
    # Look for label translation or return original key
    return axis_labels.get(key, key)


class GalvanostatRun():
    """
    Electrochemical experiment cycling on one channel.  Galvanostatic
    control potential limited (GPLC). Mass is assumed to be in grams.
    """
    cycles = []

    def __init__(self, filename, mass=None, *args, **kwargs):
        self.filename = filename
        path, ext = os.path.splitext(filename)
        file_readers = {
            '.mpr': biologic.MPRFile,
            '.mpt': biologic.MPTFile,
        }
        if ext in file_readers.keys():
            FileReader = file_readers[ext]
        else:
            msg = "Unrecognized format {}".format(ext)
            raise exceptions.FileFormatError(msg)
        # self.load_csv(filename)
        run = FileReader(filename)
        self._df = run.dataframe
        self.cycles = []
        # Remove the initial resting period
        restingIndexes = self._df.loc[self._df['mode'] == 3].index
        self._df.drop(restingIndexes, inplace=True)
        # Get theoretical capacity from eclab file
        self.theoretical_capacity = self.capacity_from_file()
        # Get currents from eclab file
        try:
            currents = self.currents_from_file()
            self.charge_current, self.discharge_current = currents
        except exceptions.ReadCurrentError:
            pass
        # Calculate capacity from charge and mass
        if mass:
            # User provided the mass
            self.mass = mass
        else:
            # Get mass from eclab file
            self.mass = run.active_mass()
        mass_g = default_units.mass(self.mass).num
        self._df.loc[:, 'capacity'] = self._df.loc[:, '(Q-Qo)/mA.h'] / mass_g
        # Process other metadata
        self.start_time = run.metadata.get('start_time', None)
        # Split the data into cycles, except the initial resting phase
        cycles = list(self._df.groupby('cycle number'))
        # Create Cycle objects for each cycle
        for cycle in cycles:
            new_cycle = Cycle(cycle[0], cycle[1])
            self.cycles.append(new_cycle)
        super().__init__(*args, **kwargs)

    # def load_csv(self, filename, *args, **kwargs):
    #     """Wrapper around pandas read_csv that filters out crappy data"""
    #     # Determine start of data
    #     with open(filename, encoding='latin-1') as dataFile:
    #         # The second line states how long the header is
    #         headerLength = int(dataFile.readlines()[1][18:20]) - 1
    #     # Skip all the initial metadata
    #     df = pd.read_csv(filename,
    #                      *args,
    #                      skiprows=headerLength,
    #                      na_values='XXX',
    #                      sep='\t',
    #                      **kwargs)
    #     self._df = df
    #     return df

    # def mass_from_file(self):
    #     """Read the mpt file and extract the sample mass"""
    #     regexp = re.compile('^Mass of active material : ([0-9.]+) ([kmµ]?g)')
    #     mass = None
    #     with open(self.filename, encoding='latin-1') as f:
    #         for line in f:
    #             match = regexp.match(line)
    #             if match:
    #                 mass_num, mass_unit = match.groups()
    #                 # We found the match, now save it
    #                 mass = units.unit(mass_unit)(float(mass_num))
    #                 break
    #     return mass

    def capacity_from_file(self):
        """Read the mpt file and extract the theoretical capacity."""
        regexp = re.compile('^for DX = [0-9], DQ = ([0-9.]+) ([kmµ]?A.h)')
        capacity = None
        with open(self.filename, encoding='latin-1') as f:
            for line in f:
                match = regexp.match(line)
                if match:
                    cap_num, cap_unit = match.groups()
                    cap_unit = cap_unit.replace('.', '')
                    # We found the match now save it
                    capacity = units.unit(cap_unit)(float(cap_num))
        return capacity

    def currents_from_file(self):
        """Read the mpt file and extract the theoretical capacity."""
        current_regexp = re.compile('^Is\s+[0-9.]+\s+([-0-9.]+)\s+([-0-9.]+)')
        unit_regexp = re.compile(
            '^unit Is\s+[kmuµ]?A\s+([kmuµ]?A)\s+([kmuµ]?A)'
        )
        data_found = False
        with open(self.filename, encoding='latin-1') as f:
            for line in f:
                # Check if this line has either the currents or the units
                current_match = current_regexp.match(line)
                unit_match = unit_regexp.match(line)
                if current_match:
                    charge_num, discharge_num = current_match.groups()
                    charge_num = float(charge_num)
                    discharge_num = float(discharge_num)
                if unit_match:
                    charge_unit, discharge_unit = unit_match.groups()
                    data_found = True
        if data_found:
            charge_current = units.unit(charge_unit)(charge_num)
            discharge_current = units.unit(discharge_unit)(discharge_num)
            return charge_current, discharge_current
        else:
            # Current data could not be extracted from file
            msg = "Could not read currents from file {filename}."
            msg = msg.format(filename=self.filename)
            raise exceptions.ReadCurrentError(msg)

    def discharge_capacity(self, cycle_idx=-1):
        """
        Return the discharge capacity of the given cycle (default last).
        """
        return self.cycles[cycle_idx].discharge_capacity()

    def charge_capacity(self, cycle_idx=-1):
        """
        Return the charge capacity of the given cycle (default last).
        """
        return self.cycles[cycle_idx].charge_capacity()

    def closest_datum(self, value, label):
        """Retrieve the datapoint that is closest to the given value along the
        given label. Works best for linear columns, like time."""
        df = self._df
        distance = (df[label] - value).abs()
        idx = df.iloc[distance.argsort()].first_valid_index()
        series = df.ix[idx]
        return series

    def plot_cycles(self, xcolumn='capacity', ycolumn='Ewe/V',
                    ax=None, *args, **kwargs):
        """
        Plot each electrochemical cycle. Additional arguments gets passed
        on to matplotlib's plot function.
        """
        if not ax:
            ax = new_axes()
        ax.set_xlabel(axis_label(xcolumn))
        ax.set_ylabel(axis_label(ycolumn))
        legend = []
        for cycle in self.cycles:
            ax = cycle.plot_cycle(xcolumn, ycolumn, ax, *args, **kwargs)
            legend.append(cycle.number)
        ax.legend(legend)
        return ax

    def plot_state_of_charge(self, framesets, ax, text="",
                             timezone='US/Central', convert_to="capacity"):
        """Plot an horizontal box with the state of charge based on the range
        of timestamps in the operando framesets. "text" will be
        plotted at the top of the box.
        """
        starttime = min([fs.starttime() for fs in framesets])
        endtime = max([fs.endtime() for fs in framesets])
        tzinfo = pytz.timezone(timezone)
        charge_start_time = self.start_time.replace(tzinfo=tzinfo)
        timemin = (starttime - charge_start_time).total_seconds()
        timemax = (endtime - charge_start_time).total_seconds()

        # Convert units from time to capacity
        capmin = self.closest_datum(value=timemin, label="time/s")[convert_to]
        capmax = self.closest_datum(value=timemax, label="time/s")[convert_to]

        # Plot a box highlighting the range of capacities
        artist = ax.axvspan(capmin,
                            capmax,
                            zorder=1,
                            facecolor="green",
                            alpha=0.15)

        # Add text label
        x = (capmin + capmax) / 2
        ylim = ax.get_ylim()
        y = ylim[1] - 0.03 * (ylim[1] - ylim[0])
        ax.text(x, y, text, horizontalalignment="center",
                verticalalignment="top")
        return artist

    def plot_discharge_capacity(self, ax=None, ax2=None):
        if not ax:
            ax = new_axes()
        if not ax2:
            ax2 = ax.twinx()
        cycle_numbers = []
        capacities = []
        efficiencies = []
        # Calculate relevant plotting values
        for cycle in self.cycles:
            cycle_numbers.append(cycle.number)
            capacities.append(cycle.discharge_capacity())
            discharge = cycle.discharge_capacity()
            charge = cycle.charge_capacity()
            efficiency = 100 * discharge / charge
            efficiencies.append(efficiency)
        ax.plot(cycle_numbers,
                capacities,
                marker='o',
                linestyle='--',
                label="Discharge capacity")
        ax2.plot(cycle_numbers,
                 efficiencies,
                 marker='^',
                 linestyle='--',
                 label="Coulombic efficiency")
        # Format axes
        if max(cycle_numbers) < 20:
            # Only show all of the ticks if there are less than 20
            ax.set_xticks(cycle_numbers)
        ax.set_xlim(0, 1 + max(cycle_numbers))
        ax.set_ylim(0, 1.1 * max(capacities))
        ax.set_xlabel('Cycle')
        ax.set_ylabel('Discharge capacity $/mAhg^{-1}$')
        ax.legend(loc='lower left')
        ax2.set_ylim(0, 105)
        ax2.legend(loc='lower right')
        ax2.set_ylabel('Coulombic efficiency (%)')
        return ax, ax2

    def plot_differential_capacity(self, ax=None, ax2=None, cycle=None):
        if not ax:
            ax = new_axes()
        if not ax2:
            ax2 = ax.twiny()
        # Let the user specify a cycle by index
        if cycle is None:
            cycles = self.cycles
        else:
            cycles = [self.cycles[cycle]]
        for cycle in cycles:
            # Plot regular charge/discharge curve
            cycle.plot_cycle(xcolumn='capacity',
                             ycolumn='Ewe/V',
                             ax=ax)
            # Plot differential capacity (sideways)
            cycle.plot_cycle(xcolumn='d(Q-Qo)/dE/mA.h/V',
                             ycolumn='Ewe/V',
                             ax=ax2,
                             linestyle='-.')
        # Annotate plot
        ax.set_xlabel('Capacity $/mAh\ g^{-1}$')
        ax2.set_xlabel('Capacity differential $/mAh\ g^{-1}V^{-1}$')
        ax.set_ylabel('Cathode potential vs $ Li/Li^+$')
        return ax, ax2
