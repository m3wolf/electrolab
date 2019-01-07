# -*- coding: utf-8 -*-

from collections import namedtuple
import math

from .exceptions import UnitCellError
from .reflection import hkl_to_tuple


class UnitCell():
    """Describes a crystallographic unit cell for XRD Refinement. Composed
    of up to three lengths (a, b and c) in angstroms and three angles
    (alpha, beta, gamma) in degrees. Subclasses with high symmetry
    with have less than six parameters.
    """
    free_parameters = ('a', 'b', 'c', 'alpha', 'beta', 'gamma')
    a = 1
    b = 1
    c = 1
    constrained_length = 1
    alpha = 90
    beta = 90
    gamma = 90

    def __init__(self, a=None, b=None, c=None,
                 alpha=None, beta=None, gamma=None):
        # Set initial cell parameters.
        # This method avoids setting constrained values to defaults
        for attr in ['a', 'b', 'c', 'alpha', 'beta', 'gamma']:
            value = locals()[attr]
            if value is not None:
                self.__setattr__(attr, value)

    def __setattr__(self, name, value):
        """Check for reasonable value for crystallography parameters"""
        # Unit cell lengths
        if name in ['a', 'b', 'c'] and value <= 0:
            msg = 'unit-cell dimensions must be greater than 0 ({}={})'
            raise UnitCellError(msg.format(name, value))
        # Unit cell angles
        elif name in ['alpha', 'beta', 'gamma'] and not (0 < value < 180):
            msg = 'unit-cell angles must be between 0° and 180° ({}={}°)'
            raise UnitCellError(msg.format(name, value))
        # No problems, so set the attribute as normal
        else:
            super(UnitCell, self).__setattr__(name, value)

    def __repr__(self):
        name = '<{cls}: a={a}, b={b}, c={c}, α={alpha}, β={beta}, γ={gamma}>'
        name = name.format(cls=self.__class__.__name__,
                           a=self.a, b=self.b, c=self.c,
                           alpha=self.alpha, beta=self.beta, gamma=self.gamma)
        return name

    def as_tuple(self):
        params = (self.a, self.b, self.c,
                  self.alpha, self.beta, self.gamma)
        return params

    @property
    def cell_parameters(self):
        """Named tuple of the cell parameters that aren't fixed."""
        params = self.free_parameters
        CellParameters = namedtuple('CellParameters', params)
        paramArgs = {param: getattr(self, param) for param in params}
        parameters = CellParameters(**paramArgs)
        return parameters

    @property
    def all_parameters(self):
        """Named tuple of all cell parameters."""
        params = ('a', 'b', 'c', 'alpha', 'beta', 'gamma')
        CellParameters = namedtuple('CellParameters', params)
        paramArgs = {param: getattr(self, param) for param in params}
        parameters = CellParameters(**paramArgs)
        return parameters

    def set_cell_parameters_from_list(self, parameters_list):
        """
        Accept a list of parameters and assumes they are in the order of
        self.free_parameters. This is a shaky assumption and should not be
        used if avoidable. This method was created for use in cell refinement
        where scipy.optimize.minimize passes a numpy array.
        """
        for idx, key in enumerate(self.free_parameters):
            setattr(self, key, parameters_list[idx])

    class FixedAngle():
        """A Unit-cell angle that cannot change for that unit cell"""
        def __init__(self, angle, name='angle'):
            self.angle = angle
            self.name = name
        
        def __get__(self, obj, objtype):
            return self.angle
        
        def __set__(self, obj, value):
            # Raise an error if the caller is trying to set a different value
            if value != self.angle:
                msg = "{name} must equal {angle}° for {cls} (got {value}°)"
                msg = msg.format(name=self.name,
                                 angle=self.angle,
                                 cls=obj.__class__.__name__,
                                 value=value)
                raise UnitCellError(msg)
    
    class ConstrainedLength():
        """
        Unit-cell angle that is tied to another length in the cell. Eg. a=b
        """
        def __get__(self, obj, objtype):
            return obj.constrained_length
        
        def __set__(self, obj, value):
            obj.constrained_length = value


class CubicUnitCell(UnitCell):
    """Unit cell where a=b=c and α=β=γ=90°"""
    free_parameters = ('a', )
    # Descriptors for unit-cell lengths, since a=b=c
    a = UnitCell.ConstrainedLength()
    b = UnitCell.ConstrainedLength()
    c = UnitCell.ConstrainedLength()
    alpha = UnitCell.FixedAngle(90, name="α")
    beta = UnitCell.FixedAngle(90, name="β")
    gamma = UnitCell.FixedAngle(90, name="γ")
    
    def d_spacing(self, hkl):
        """Determine d-space for the given hkl plane."""
        h, k, l = hkl_to_tuple(hkl)
        inverse_d_squared = (h**2 + k**2 + l**2) / (self.a**2)
        d = math.sqrt(1 / inverse_d_squared)
        return d


class HexagonalUnitCell(UnitCell):
    """Unit cell where a=b, α=β=90°, γ=120°."""
    free_parameters = ('a', 'c')
    a = UnitCell.ConstrainedLength()
    b = UnitCell.ConstrainedLength()
    alpha = UnitCell.FixedAngle(angle=90, name="α")
    beta = UnitCell.FixedAngle(angle=90, name="β")
    gamma = UnitCell.FixedAngle(angle=120, name="γ")
    
    def d_spacing(self, hkl):
        """Determine d-space for the given hkl plane."""
        h, k, l = hkl_to_tuple(hkl)
        a, c = (self.a, self.c)
        inverse_d_squared = 4 * (h**2 + h * k + k**2) / (3 * a**2) + l**2 / (c**2)
        d = math.sqrt(1 / inverse_d_squared)
        return d


class TetragonalUnitCell(UnitCell):
    """Unit cell where a=b, α=β=γ=90°."""
    free_parameters = ('a', 'c')
    a = UnitCell.ConstrainedLength()
    b = UnitCell.ConstrainedLength()
    alpha = UnitCell.FixedAngle(angle=90, name="α")
    beta = UnitCell.FixedAngle(angle=90, name="β")
    gamma = UnitCell.FixedAngle(angle=90, name="γ")

    def d_spacing(self, hkl):
        """Determine d-space for the given hkl plane."""
        h, k, l = hkl_to_tuple(hkl)
        a, c = (self.a, self.c)
        inverse_d_squared = (h**2 + k**2) / (a**2) + l**2 / (c**2)
        d = math.sqrt(1 / inverse_d_squared)
        return d
