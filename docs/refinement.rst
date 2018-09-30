============
 Refinement
============

Analyzing XRD mapping data requires some sort of refinement on each
mapping position. To accomplish this, the scimap provides the
:py:meth:`scimap.xrd_map.refine_mapping_data` method. Once refined,
the results are stored in the HDF5 file and can be plotting using
:py:meth:`scimap.xrd_map.plot_map`. Different approaches are activated
using the ``backend`` parameter. Many of the options, however, are
unfinished or imperfect:

Fullprof
========

.. warning:: This backend is functional but fragile.

This backend requires that `FullProf`_ refinement be installed and
available. The environmental variable ``$FULLPROF`` should point to
the installation directory.

.. _FullProf: https://www.ill.eu/sites/fullprof/
	     
.. code:: python

    mymap = scimap.XRDMap(...)
    mymap.refine_mapping_data(backend="fullprof")

FullProf refinement creates temporary files that are cleaned up upon
successful refinement; if refinement fails, these files will be left
behind for troubleshooting.

GSAS-II
=======

.. error:: This backend is not functional. GSAS-II is under active
           development and if an API for Pawley refinement becomes
           available, this backend may be updated.

Pawley
======

.. warning:: This backend is incomplete. Us at your own risk.

This backend is an implementation of simple Pawley refinement in
python.

Custom-Backends
===============

If none of the available backends suit your needs, a custom backend
may be provided. The backend should be a subclass of
:py:class:`scimap.base_refinement.BaseRefinement`. The
:py:meth:`~scimap.base_refinement.BaseRefinement.predict` method
should return the predicted intensities based on 2θ values, and a
number of methods should be overridden that accept 2θ values and
return refined parameters:

- :py:meth:`~scimap.base_refinement.BaseRefinement.goodness_of_fit`
- :py:meth:`~scimap.base_refinement.BaseRefinement.background`
- :py:meth:`~scimap.base_refinement.BaseRefinement.cell_params`
- :py:meth:`~scimap.base_refinement.BaseRefinement.scale_factor`
- :py:meth:`~scimap.base_refinement.BaseRefinement.broadenings`
- :py:meth:`~scimap.base_refinement.BaseRefinement.phase_fractions`

.. code:: python

    # Sub-class the base refinement
    class CustomRefinement(scimap.BaseRefinement):
        def phase_fractions(self, two_theta, intensities):
	    # Do some calculations here
	    ...
	# Override the other methods here
	...

    # Now do the refinement
    mymap = scimap.XRDMap(...)
    mymap.refine_mapping_data(backend=CustomRefinement)

