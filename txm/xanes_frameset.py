import os
from collections import defaultdict
import re
import math

import pandas as pd
from matplotlib import pyplot
from matplotlib.colors import Normalize
import h5py
import numpy as np

from utilities import display_progress
from .frame import TXMFrame, average_frames
from .gtk_viewer import GtkTxmViewer
import exceptions
from hdf import HDFAttribute


def build_dataframe(frames):
    index = [frame.energy for frame in frames]
    images = [pd.DataFrame(frame.image_data) for frame in frames]
    series = pd.Series(images, index=index)
    return series

class XanesFrameset():
    _attrs = {}
    active_groupname = HDFAttribute('active_groupname')
    background_groupname = HDFAttribute('background_groupname')

    def __init__(self, filename, groupname):
        self.hdf_filename = filename
        self.group_name = groupname
        # Check to make sure a valid group is given
        with self.hdf_file() as hdf_file:
            if not groupname in hdf_file.keys():
                msg = "Created new frameset group: {}"
                print(msg.format(groupname))
                hdf_file.create_group(groupname)

    def __iter__(self):
        """Get each frame from the HDF5 file"""
        hdf_file = self.hdf_file()
        for dataset_name in self.active_group().keys():
            yield TXMFrame.load_from_dataset(self.active_group()[dataset_name])

    def __len__(self):
        return len(self.active_group().keys())

    def __getitem__(self, index):
        hdf_file = self.hdf_file()
        dataset_name = list(self.active_group().keys())[index]
        return TXMFrame.load_from_dataset(self.active_group()[dataset_name])

    def switch_group(self, name):
        self.active_groupname = name

    def fork_group(self, name):
        # Create an empty group
        try:
            del self.hdf_group()[name]
        except KeyError:
            pass
        # Copy the old data
        self.hdf_group().copy(self.active_groupname, name)
        # Update the group name
        self.active_groupname = name

    def subtract_background(self, bg_groupname):
        self.background_groupname = bg_groupname
        self.fork_group('absorbance_frames')
        bg_group = self.hdf_file()[bg_groupname]
        for energy in display_progress(self.active_group().keys(), "Subtracting background"):
            sample_dataset = self.active_group()[energy]
            bg_dataset = bg_group[energy]
            new_data = np.log10(bg_dataset.value/sample_dataset.value)
            sample_dataset.write_direct(new_data)

    def align_frame_positions(self):
        """Correct for inaccurate motion in the sample motors."""
        self.fork_group('aligned_frames')
        # Determine average positions
        total_x = 0; total_y = 0; n=0
        for frame in display_progress(self, 'Computing true center'):
            n+=1
            total_x += frame.sample_position.x
            total_y += frame.sample_position.y
        global_x = total_x / n
        global_y = total_y / n
        for frame in display_progress(self, 'Aligning frames'):
            um_per_pixel_x = 40/frame.image_data.shape[1]
            um_per_pixel_y = 40/frame.image_data.shape[0]
            offset_x = int(round((global_x - frame.sample_position.x)/um_per_pixel_x))
            offset_y = int(round((global_y - frame.sample_position.y)/um_per_pixel_y))
            frame.shift_data(x_offset=offset_x, y_offset=offset_y)
            # Store updated position info
            new_position = (
                frame.sample_position.x + offset_x * um_per_pixel_x,
                frame.sample_position.y + offset_y * um_per_pixel_y,
                frame.sample_position.z
            )
            frame.sample_position = new_position

    def label_particles(self):
        # Create a new group
        if 'particle_labels' in self.hdf_group().keys():
            del self.hdf_group()['particle_labels']
        labels_group = self.hdf_group().create_group('particle_labels')
        for frame in display_progress(self, 'Identifying particles'):
            key = frame.image_data.name.split('/')[-1]
            data = frame.calculate_particle_labels()
            dataset = labels_group.create_dataset(name=key, data=data)
            frame.particle_labels_path = dataset.name

    def plot_full_image(self):
        return pyplot.imshow(self.df.mean())

    def xanes_spectrum(self):
        """Collapse the dataset down to a two-d spectrum."""
        energies = []
        intensities = []
        bg_group = self.background_group()
        for frame in self:
            energies.append(frame.energy)
            # Sum absorbances for datasets
            intensity = np.sum(frame.image_data)/np.prod(frame.image_data.shape)
            intensities.append(intensity)
        # Combine into a series
        series = pd.Series(intensities, index=energies)
        return series

    def plot_xanes_spectrum(self):
        spectrum = self.xanes_spectrum()
        ax = spectrum.plot()
        ax.set_xlabel('Energy /eV')
        ax.set_ylabel('Overall absorbance')
        return ax

    def hdf_file(self):
        # Determine filename
        try:
            file = h5py.File(self.hdf_filename, 'r+')
        except OSError as e:
            # HDF File does not exist, make a new one
            print('Creating new HDF5 file: {}'.format(self.hdf_filename))
            file = h5py.File(self.hdf_filename, 'w-')
            file.create_group(self.group_name)
        return file

    def hdf_group(self):
        return self.hdf_file()[self.group_name]

    def background_group(self):
        return self.hdf_file()[self.background_groupname]

    @property
    def hdf_node(self):
        """For use with HDFAttribute descriptor."""
        return self.hdf_group()

    def active_group(self):
        parent_group = self.hdf_group()
        if self.active_groupname is None:
            group = parent_group
        else:
            if not self.active_groupname in parent_group.keys():
                # Create group if necessary
                parent_group.create_group(self.active_groupname)
            group = parent_group[self.active_groupname]
        return group

    def add_frame(self, frame):
        setname_template = "{energy}_eV{serial}"
        frames_group = self.active_group()
        # Find a unique frame dataset
        setname = setname_template.format(
            energy=frame.approximate_energy,
            serial=""
        )
        counter = 0
        while setname in frames_group.keys():
            counter += 1
            setname = setname_template.format(
                energy=frame.approximate_energy,
                serial="-" + str(counter)
            )
        # Name found, so create the actual dataset
        new_dataset = frame.create_dataset(setname=setname,
                                           hdf_group=frames_group)
        return setname

    def normalizer(self):
        # Find global limits
        global_min = 0
        global_max = 99999999999
        for frame in self:
            local_min = np.min(frame.image_data)
            if local_min < global_min:
                global_min = local_min
            local_max = np.max(frame.image_data)
            if local_max < global_max:
                global_max = local_max
        return Normalize(global_min, global_max)

    def gtk_viewer(self):
        viewer = GtkTxmViewer(frameset=self)
        viewer.show()
