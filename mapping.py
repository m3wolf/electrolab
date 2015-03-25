# -*- coding: utf-8 -*-

import jinja2, math
from matplotlib import pylab, pyplot, collections, patches, colors, cm
import numpy as np
import pandas as pd
import os

def new_axes():
    """Create a new set of matplotlib axes for plotting"""
    height = 5 # in inches
    # Adjust width to accomodate colorbar
    width = height/0.8
    fig = pyplot.figure(figsize=(width, height))
    ax = pyplot.gca()
    return ax

class Cube():
    """Cubic coordinates of a hexagon"""
    def __init__(self, i, j, k, *args, **kwargs):
        self.i = i
        self.j = j
        self.k = k
        super(Cube, self).__init__(*args, **kwargs)

    def __getitem__(self, key):
        coord_list = [self.i, self.j, self.k]
        return coord_list[key]

    def __add__(self, other):
        new = Cube(
            self.i + other.i,
            self.j + other.j,
            self.k + other.k,
        )
        return new


    def __eq__(self, other):
        result = False
        if self.i == other.i and self.j == other.j and self.k == other.k:
            result = True
        return result

    def __str__(self):
        return "({i}, {j}, {k})".format(i=self.i, j=self.j, k=self.k)

    def __repr__(self):
        return "Cube{0}".format(self.__str__())


class BaseSample():
    """
    A physical sample that gets mapped by XRD, presumed to be circular
    with center and diameter in millimeters. Collimator size given in mm.
    """
    cmap_name = 'autumn'
    two_theta_range = (50, 90) # Detector angle range in degrees
    THETA1_MIN=0 # Source limits based on geometry
    THETA1_MAX=50
    THETA2_MIN=0 # Detector limits based on geometry
    THETA2_MAX=55
    scan_time = 3 # Seconds at each detector angle
    frame_step = 20 # How much to move detector by in degrees
    frame_width = 30 # 2-theta coverage of detector face
    scans = []
    def __init__(self, center, diameter, collimator=0.5, rows=None,
                 sample_name='unknown', *args, **kwargs):
        self.center = center
        self.diameter = diameter
        # Determine number of rows from collimator size
        if rows is None:
            self.rows = math.ceil(diameter/collimator/2)
        else:
            self.rows = rows
        self.sample_name = sample_name
        self.create_scans()
        return super(BaseSample, self).__init__(*args, **kwargs)

    @property
    def unit_size(self):
        return self.diameter / self.rows / math.sqrt(3)

    def create_scans(self):
        """Populate the scans array with new scans in a hexagonal array."""
        self.scans = []
        for idx, coords in enumerate(self.path(self.rows)):
            filename = 'map-{n:x}'.format(
                sample=self.sample_name,
                n=idx
            )
            new_scan = self.XRDScan(coords, filename, sample=self)
            self.scans.append(new_scan)

    def scan(self, cube):
        """Find a scan in the array give a set of cubic coordinates"""
        result = None
        for scan in self.scans:
            if scan.cube_coords == cube:
                result = scan
                break
        return result

    def path(self, rows):
        """Generator gives coordinates for a spiral path around the sample."""
        # Six different directions one can move
        basis_set = {
            'W': Cube(-1, 1, 0),
            'SW': Cube(-1, 0, 1),
            'SE': Cube(0, -1, 1),
            'E': Cube(1, -1, 0),
            'NE': Cube(1, 0, -1),
            'NW': Cube(0, 1, -1)
        }
        # Start in the center
        curr_coords = Cube(0, 0, 0)
        yield curr_coords
        # Spiral through each row
        for row in range(1, rows+1):
            # Move to next row
            curr_coords += basis_set['NE']
            yield curr_coords
            for i in range(0, row-1):
                curr_coords += basis_set['NW']
                yield curr_coords
            # Go around the ring for each basis vector
            for key in ['W', 'SW', 'SE', 'E', 'NE']:
                vector = basis_set[key]
                for i in range(0, row):
                    curr_coords += vector
                    yield curr_coords

    def write_slamfile(self, f=None):
        """
        Format the sample into a slam file that GADDS can process.
        """
        # Import template
        env = jinja2.Environment(loader=jinja2.PackageLoader('electrolab', ''))
        template = env.get_template('mapping-template.slm')
        self.create_scans()
        context = self.get_context()
        # Create file and directory if necessary
        if f is None:
            directory = '{samplename}-frames'.format(
                samplename=self.sample_name
            )
            if not os.path.exists(directory):
                os.makedirs(directory)
            filename = '{dir}/{samplename}.slm'.format(
                dir=directory, samplename=self.sample_name
            )
            with open(filename, 'w') as f:
                f.write(template.render(**context))
        else:
            f.write(template.render(**context))
        return f

    def get_context(self):
        """Convert the object to a dictionary for the templating engine."""
        # Estimate the total time
        secs = len(self.scans)*self.scan_time
        total_time = "{0}s ({1:0.1f}h)".format(secs, secs/3600)
        # List of frames to integrate
        frames = []
        for frame_num in range(0, self.get_number_of_frames()):
            start = self.two_theta_range[0] + 2.5 + frame_num*self.frame_step
            end = start + self.frame_step
            frame = {
                'start': start,
                'end': end,
                'number': frame_num,
            }
            frames.append(frame)
        context = {
            'scans': [],
            'num_scans': len(self.scans),
            'frames': frames,
            'frame_step': self.frame_step,
            'number_of_frames': self.get_number_of_frames(),
            'xoffset': self.center[0],
            'yoffset': self.center[1],
            'theta1': self.get_theta1(),
            'theta2': self.get_theta2_start(),
            'scan_time': self.scan_time,
            'total_time': total_time,
            'sample_name': self.sample_name
        }
        for idx, scan in enumerate(self.scans):
            # Prepare scan-specific details
            x, y = scan.xy_coords(unit_size=self.unit_size)
            scan_metadata = {'x': x, 'y': y, 'filename': scan.filename}
            context['scans'].append(scan_metadata)
        return context

    def get_number_of_frames(self):
        theta1 = self.get_theta1()
        num_frames = math.ceil(
            (self.two_theta_range[1]-self.two_theta_range[0])/self.frame_step
        )
        # Check for values outside instrument limits
        t2_start = self.get_theta2_start()
        t2_end = t2_start + num_frames*self.frame_step
        if t2_end > self.THETA2_MAX:
            msg = "2-theta range {given} is outside detector limits: {limits}".format(
                given=self.two_theta_range,
                limits=(self.THETA2_MIN, self.THETA2_MAX))
            raise ValueError(msg)
        return num_frames

    def get_theta2_start(self):
        # Assuming that theta1 starts at highest possible range
        theta1 = self.get_theta1()
        theta2_bottom = self.two_theta_range[0] - theta1
        theta2_start = theta2_bottom - self.frame_width/8 + self.frame_width/2
        return theta2_start

    def get_theta1(self):
        # Check for values outside preset limits
        theta1 = self.two_theta_range[0]
        if theta1 < self.THETA1_MIN:
            msg = "2-theta range {given} is outside source limits: {limits}".format(
                given=self.two_theta_range,
                limits=(self.THETA1_MIN, self.THETA1_MAX))
            raise ValueError(msg)
        elif theta1 > self.THETA1_MAX:
            # Cap the theta1 value at a safety limited maximum
            theta1 = self.THETA1_MAX
        return theta1

    def get_cmap(self):
        """Return a function that converts values in range 0 to 1 to colors."""
        return pyplot.get_cmap(self.cmap_name)

    def plot_map(self, ax=None):
        x = []
        y = []
        values = []
        colors = []
        i = 0
        cmap = self.get_cmap()
        for scan in self.scans:
            i+=1
            coord = scan.xy_coords(self.unit_size)
            x.append(coord[0])
            y.append(coord[1])
            metric = scan.metric()
            values.append(metric)
            if metric is None:
                # Invalid scan
                colors.append('white')
            else:
                colors.append(cmap(metric))
        xy = list(zip(x, y))
        # Build and show the hexagons
        if not ax:
            # New axes unless one was already created
            ax = new_axes()
        xy_lim = self.diameter/2*1.25
        ax.set_xlim([-xy_lim, xy_lim])
        ax.set_ylim([-xy_lim, xy_lim])
        ax.set_xlabel('mm')
        ax.set_ylabel('mm')
        for key, loc in enumerate(xy):
            hexagon = patches.RegularPolygon(xy=loc,
                                             numVertices=6,
                                             radius=0.57*self.unit_size,
                                             color=colors[key])
            ax.add_patch(hexagon)
        # Add circle for theoretical edge
        circle = patches.Circle((0, 0), radius=self.diameter/2,
                                edgecolor='blue', fill=False, linestyle='dashed')
        ax.add_patch(circle)
        # Add colormap to the side of the axes
        mappable = cm.ScalarMappable(cmap=cmap)
        mappable.set_array(np.arange(0, 2))
        pyplot.colorbar(mappable, ax=ax)
        return ax

    def __repr__(self):
        return '<Sample: {name}>'.format(name=self.sample_name)

    class XRDScan():
        """
        An XRD scan at one X,Y location. Several Scan objects make up a
        Sample object.
        """
        def __init__(self, location, filename, sample=None, *args, **kwargs):
            self.cube_coords = location
            self.filename = filename
            self.sample = sample
            return super(BaseSample.XRDScan, self).__init__(*args, **kwargs)

        def xy_coords(self, unit_size=1):
            # Convert internal coordinates to conventional cartesian coords
            cube = self.cube_coords
            x = unit_size * 1/2 * (cube.i - cube.j)
            y = unit_size * math.sqrt(3)/2 * (cube.i + cube.j)
            return (x, y)

        def load_spectrum(self):
            filename = "{0}.plt".format(self.filename)
            df = pd.read_csv(filename, names=['2theta', 'counts'],
                             sep=' ', comment='!', index_col=0)
            return df

        def metric(self):
            """
            Returns the calculated metric. Ex: ratio of two different phases.
            Intended to be overwritten by subclasses for specific
            sample materials.
            """
            # Just return the distance from bottom left to top right
            p = self.cube_coords[0]
            rows = self.sample.rows
            r = p/2/rows + 0.5
            return r

        def plot_spectrum(self, ax=None):
            df = self.load_spectrum()
            if not ax:
                fig = pyplot.figure()
                ax = pyplot.gca()
            ax.plot(df.index, df.loc[:, 'counts'])
            ax.set_xlabel('2θ')
            ax.set_ylabel('Counts')
            title = 'XRD Spectrum at ({i}, {j}, {k})'.format(
                i=self.cube_coords[0],
                j=self.cube_coords[1],
                k=self.cube_coords[2]
            )
            ax.set_title(title)
            return ax

class LMOSample(BaseSample):
    """
    Sample for mapping LiMn2O4 cathodes.
    """
    two_theta_range = (30, 50)
    scan_time = 600 # 10 minutes per frame
    peaks_by_hkl = {
        '111': (18, 20),
        '311': (35, 38),
        '222': (37.5, 39),
        '400': (42, 47),
        '331': (47, 50),
        '333': (55, 62),
        '511': (55, 62),
        '440': (62, 67),
        '531': (67, 70),
    }
    class XRDScan(BaseSample.XRDScan):
        def metric(self):
            """
            Compare the 2θ positions of two peaks. Using two peaks may correct
            for differences is sample height on the instrument.
            """
            # Linear regression values determined by experiment
            slope = -0.161144716842
            yIntercept = 7.99106396434
            result = 0
            df = self.load_spectrum()
            # Decide on two peaks to use for comparison
            # List of possible peaks to look for
            normal_range = (0.2, 1)
            # Get the 2θ value of peak 1
            peak1 = LMOSample.peaks_by_hkl['311']
            range1 = df.loc[peak1[0]:peak1[1], 'counts']
            theta1 = range1.argmax()
            # Get the 2θ value of peak 2
            peak2 = LMOSample.peaks_by_hkl['400']
            range2 = df.loc[peak2[0]:peak2[1], 'counts']
            theta2 = range2.argmax()
            print(theta1, theta2)
            # Check for non-sample scans (background tape, etc)
            if df.loc[theta2, 'counts'] > 600 and df.loc[theta1, 'counts'] > 400:
                # Subtract the 2theta values of the two peaks
                diff = theta2 - theta1
                # Apply calibration curve
                result = (diff-yIntercept)/slope
                # Normalize to result to the range 0 to 1
                result = (result - normal_range[0])/(normal_range[1]-normal_range[0])
            else:
                # Some other background-type scan was collected
                result = None
            # Return the result
            return result
