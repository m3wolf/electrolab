===============================
 Accessing the Underlying Data
===============================

Scimap tries to keep controller and presentation logic separate from
the underlying data; most of the methods on the ``XRDMap`` class
retrieve, display and/or store their data without the user's
involvement. Given the complicated nature of scientific analysis, it
can sometimes become necessary to retrieve individual diffraction
patterns directly, or even to manipulate the data files directly;
scimap provides a mechanism for both cases.

Scimap uses HDF5 files to store all the mapping data. This provides
two benefits: 1) large datasets can still be analyzed even if they
don't fit into main memory, and 2) the results of analysis can easily
be shared or published as one file with accompanying metadata. This
comes at the cost of increased time needed to write calculated data to
disk, rather than manipulating it in main memory.

Retrieving Common Representations
=================================

The following methods of the ``XRDMap`` class can be used to retrieve
a variety of packaged data. 

- XRDMap.diffractogram: Get bulk-averaged diffraction data from all
  positions.
- XRDMap.get_diffractogram: Get the diffraction data for a single
  position.

Accessing the XRD Data Store and HDF File
=========================================

It is possible to interact with the data as numerical arrays. The
``XRDStore`` class provides an interface for accessing the defined
datasets. It can be retrieved throught the ``XRDMap().store()``
method, or instantiated directly.

.. warning:: The XRDStore class should be used as a **context manager**
             whenever possible. Failure to close the underlying HDF5
             file, especially if using a writeable mode, is likely to
             lead to file corruption.

.. code:: python

   xmap = XRDMap(...)
   with xmap.store() as store:
       Is = store.intensities

In the above example, ``store.intensities`` gives the intensity
(photon counts) for each mapping position. The result will be an `m x n`
array where `m` is the number of mapping positions and `n` is the number
of angles/scattering vectors.
