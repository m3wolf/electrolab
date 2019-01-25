# -*- coding: utf-8 -*-

import copy
from collections import namedtuple
import re

from . import exceptions

HKL = namedtuple('HKL', ('h', 'k', 'l'))


def hkl_to_tuple(hkl_input):
    """If hkl_input is a string, extract the hkl values and
    return them as (h, k, l). If hkl_string is not a string, return it
    unmodified."""
    # Convert hkl to tuple dependent on form
    hklTuple = None
    if not isinstance(hkl_input, str):
        # Already a tuple, no action
        hklTuple = hkl_input
    elif len(hkl_input) == 4:
        # Vaguly formed index eg (1010)
        msg = 'Ambiguous hkl value "{}". Use spaces or commas as separators.'
        msg = msg.format(hkl_input)
        raise exceptions.HKLFormatError(msg)
    elif len(hkl_input) == 3:
        # Simple (hkl) string to tuple
        hkl = [int(x) for x in hkl_input]
        hklTuple = HKL(*hkl)
    else:
        # The index probably contains separators
        regex = r'(\d+)[, ](\d+)[, ](\d+)'
        result = re.match(regex, hkl_input)
        if not result:
            msg = 'Malformed hkl value "{}".'.format(hkl_input)
            raise exceptions.HKLFormatError(msg)
        hkl = [int(x) for x in result.groups()]
        hklTuple = HKL(*hkl)

    return hklTuple


class Reflection():
    """An XRD reflection with a specific hkl value.
    
    Argument
    --------
    - hkl : Tuple of h, k and l values for this refelctions crystal plane.
    - qrange : 2-tuple of scattering vector lengths that define the
      boundaries of this reflection.
    
    """
    def __init__(self, hkl=(0, 0, 0), qrange=(0.71, 5.24),
                 multiplicity=1, fwhm=0.03, intensity=100):
        h, k, l = hkl_to_tuple(hkl)
        self._h = h
        self._k = k
        self._l = l
        self.multiplicity = multiplicity
        self.intensity = intensity
        self.fwhm = 0.03
        self.qrange = qrange
    
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
    
    def copy(self):
        """Create a duplicate of this object with identifcal parameters.
        
        This can be useful for other classes that use ``Reflection()``
        objects as attributes during import. This way, reflections can
        be changed during runtime without clobbering other objects of
        the same class.
        
        See the :py:class:`scimap.phase.Phase` class for an example.
        
        """
        return copy.copy(self)
