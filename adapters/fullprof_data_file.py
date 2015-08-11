# -*- coding: utf-8 -*-
import pandas as pd

class FullProfDataFile():
    def __init__(self, filename):
        self.filename = filename

    def write_diffractogram(self, scan):
        """
        Write the 2-theta, counts data for the given scan in a format
        suitable for feeding into the FullProf refinement program.
        """
        df = scan.diffractogram
        result = df.to_csv(self.filename, columns=['counts'], sep=' ')
