import os

import unittest
import h5py

import scimap

# class ScimapTestCase(unittest.TestCase):
#     def assertApproximatelyEqual(self, actual, expected,
#                                  tolerance=0.01, msg=None):
#         """Assert that 'actual' is within relative 'tolerance' of
#         'expected'."""
#         # Check for lists
#         if isinstance(actual, list) or isinstance(actual, tuple):
#             for a, b in zip(actual, expected):
#                 self.assertApproximatelyEqual(a, b, tolerance=tolerance)
#         else:
#             diff = abs(actual - expected)
#             acceptable_diff = abs(expected * tolerance)
#             if diff > acceptable_diff:
#                 msg = "{actual} is not close to {expected}"
#                 self.fail(msg=msg.format(actual=actual, expected=expected))


class HDFTestCase(unittest.TestCase):
    """A test case that sets up and tears down an HDF file."""
    def setUp(self):
        curdir = os.path.dirname(os.path.realpath(__file__))
        self.hdf_filename = os.path.join(curdir, 'test-data-xrd', 'xrd-map-gadds.hdf')
        if not os.path.exists(self.hdf_filename):
            # Create the HDF file from real data
            scimap.write_gadds_script(qrange=(1, 5),
                                      sample_name="xrd-map-gadds", center=(0, 0),
                                      hexadecimal=False, collimator=0.8,
                                      hdf_filename=self.hdf_filename)
            scimap.import_gadds_map(directory="test-data-xrd/xrd-map-gadds/",
                                    tube="Cu", hdf_filename=self.hdf_filename,
                                    hdf_groupname='xrd-map-gadds')
            gaddsmap = scimap.XRDMap(hdf_filename=self.hdf_filename,
                                     sample_name="xrd-map-gadds", Phases=[scimap.lmo.CubicLMO])
            gaddsmap.refine_mapping_data()
        # if os.path.exists(self.hdf_filename): 
        #     os.remove(self.hdf_filename)
        # self.hdf_file = h5py.File(self.hdf_filename, 'w-')

    # def tearDown(self):
    #     self.hdf_file.close()
    #     if os.path.exists(self.hdf_filename):
    #         os.remove(self.hdf_filename)
