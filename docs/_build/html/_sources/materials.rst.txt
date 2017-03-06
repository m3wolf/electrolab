===========
 Materials
===========

Having a proper understanding of the material structure being studied
is key to extracting usable information to map. The
``scimap.XRDMap()`` class accepts a ``Phases=[]`` argument, which is a
list of ``scimap.Phase`` subclasses. The modules ``scimap.lmo`` and
``scimap.nca`` include pre-defined phases for |LMO| and |NCA|
respectively. Unless you happen to be working with these materials,
you will likely need to define some classes in order to perform a
thorough analysis.

.. |LMO| replace:: LiMn\ :sub:`2`\ O\ :sub:`4`\ 

.. |NCA| replace:: LiNi\ :sub:`0.8`\ Co\ :sub:`0.15`\ Al\ :sub:`0.05`\ O\ :sub:`2`\ 

.. code:: python

   from scimap import Phase, TetragonalUnitCell, XRDMap
   
   # Create a new class for our material
   class Unobtainium(Phase):
       unit_cell = TetragonalUnitCell()
       # Define a list of hkl planes that define this strcture
       reflection_list = [
           Reflection('000', qrange=(2.75, 2.82)),
       ]

    # Now use our new class to analyze mapping data
    mymap = XRDMap(Phases=[Unobtainium])

Notice that the `Unobtainium` class is not instantiated before being
passed to `XRDMap`. This is because each mapping position gets a new
phase object that can be refined.

The `reflection_list` attribute describes the crystallographic
reflection planes for this crystal system. Each entry in this list is
a `Reflection` object. The first argument is a string with the hkl
indices. Additionally, a `qrange` argument (2-tuple) should be given
that gives the scattering vector (q) limits. Q can be calculated from
2θ values at a given wavelength λ:

.. math::

   q = \frac{4\pi}{\lambda} sin \Big(\frac{2\theta}{2}\Big)

As a shortcut, the function ``scimap.twotheta_to_q`` can also be used.
