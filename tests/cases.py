import unittest

class ScimapTestCase(unittest.TestCase):
    def assertApproximatelyEqual(self, actual, expected, tolerance=0.01, msg=None):
        """Assert that 'actual' is within relative 'tolerance' of 'expected'."""
        # Check for lists
        if isinstance(actual, list) or isinstance(actual, tuple):
            for a, b in zip(actual, expected):
                self.assertApproximatelyEqual(a, b, tolerance=tolerance)
        else:
            diff = abs(actual-expected)
            acceptable_diff = abs(expected * tolerance)
            if diff > acceptable_diff:
                msg = "{actual} is not close to {expected}"
                self.fail(msg=msg.format(actual=actual, expected=expected))
