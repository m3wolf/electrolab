======================================
 Instrumentation and Beamline Support
======================================

Bruker GADDS Diffractometers
============================

Bruker has a software package that is shipped with "Series II"
diffractometers, called "GADDS". While this software is being replaced
by an overhauled suite, it does have a scripting language that allows
for control of the instrument via scimap.

Use of Bruker GADDS system with scimap has three main components

Data Acquisition Script
-----------------------

This is where the user will prepare a "slm file" which can be run in
GADDS. It will instruct the instrument to trace out a desired path and
save both 2D and 1D diffractograms, as well as a .jpg of the camera's
view.

.. code:: python

    scimap.write_gadds_script(qrange=(1, 5), sample_name='example',
                              center=(1.32, -44.78), collimator=0.8)

This will create an ``example-frames/example.slm`` file that can
be run using GADDS. It will also create some supporting files that
will be used in the subsequent analysis.

Importing and Pre-Processing
----------------------------

Once the GADDS script has been run, the data can then be imported into
the HDF file for further analysis.

.. code:: python

	  scimap.import_gadds_map(sample_name="example",
	                          directory="example_frames")

The value for ``sample_name`` should match that passed to the
``write_gadds_script`` function called above. After calling this
function, the file ``example.h5`` contains the imported data.

Analysis and Visualization
--------------------------

Coming soon.

Bruker DaVinci Diffractometers
==============================

As of this writing, the modern Bruker diffractometer family does not
support scripting and so mapping through scimap is not
possible. Support for individual Bruker BRML files is available
through ``scimap.XRDScan`` objects.

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
automatically converted to scattering length (*q*).
