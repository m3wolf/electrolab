# -*- coding: utf-8 -*-

from collections import namedtuple

HKL = namedtuple('HKL', ('h', 'k', 'l'))

def hkl_to_tuple(hkl_input):
    """If hkl_input is a string, extract the hkl values and
    return them as (h, k, l). If hkl_string is not a string, return it
    unmodified."""
    # Convert hkl to tuple dependent on form
    hklTuple = None
    if isinstance(hkl_input, tuple):
        # Already a tuple, no action
        hklTuple = hkl_input
    elif isinstance(hkl_input, str):
        # String to tuple
        hklTuple = (
            int(hkl_input[0]),
            int(hkl_input[1]),
            int(hkl_input[2])
        )
    return hklTuple

class Reflection():
    """An XRD reflection with a specific hkl value."""
    def __init__(self, hkl=(0, 0, 0), two_theta_range=(10, 80), multiplicity=1, intensity=100):
        h, k, l = hkl_to_tuple(hkl)
        self._h = h
        self._k = k
        self._l = l
        self.multiplicity = multiplicity
        self.intensity = intensity
        self.two_theta_range = two_theta_range

    @property
    def hkl(self):
        hkl_tuple = HKL(self._h, self._k, self._l)
        return hkl_tuple

    @property
    def hkl_string(self):
        string = "{h}{k}{l}".format(h=self._h, k=self._k, l=self._l)
        return string

    def __repr__(self):
        template = "<Reflection: {0}{1}{2}>"
        return template.format(self._h, self._k, self._l)

    def __str__(self):
        template = "({0}{1}{2})"
        return template.format(self._h, self._k, self._l)
