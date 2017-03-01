=================================
 Instrument and Beamline Support
=================================

Bruker GADDS Diffractometers
============================

Bruker has a software package that is shipped with "Series II"
diffractometers. While this software has many limitations, it does
have a scripting language that allows for control of the instrument
via scimap.

Use of Bruker GADDS system with scimap has three main components

Data Acquisition
----------------

This is where the user will prepare a "slm file" which can be run in
GADDS. It will instruct the instrument to trace out a desired path and
save both 2D and 1D diffractograms. 

.. code:: python

   scimap.write_gadds_script()

Importing and Pre-Processing
----------------------------

Analysis and Visualization
--------------------------

Bruker DaVinci Diffractometers
==============================

APS Beamline 34-ID-E
====================

Scimap can import integrated data obtained from the 34-ID-E
microdiffraction beamline at the Advanced Photon Source. The
two-dimensional diffraction patterns must first be integrated to
one-dimensional patterns using the ``Fit2D`` application. Once
``.chi`` files are prepared, they can be imported into scimap.

.. code:: python

   scimap.import_aps_34IDE_map(directory='example_frames/',
                               wavelength=0.516, shape=(10, 10),
			       step_size=0.10)

Where the wavelength is given in angstroms. Ideally, the .chi files
should use scattering lengths, but any files in 20 will be
automatically converted to scattering length $q$.
