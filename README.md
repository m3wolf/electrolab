# SciMap: Scientific Mapping for Python

## Motivation

This project aims to take large datasets in chemistry and physics, and
distill them down to two-dimensional spatial maps where the color
represents some physically meaningful calculated metric.

## Graphical User Interface

Several modules provide a GUI for viewing computed data. These
sections make use of **GTK3** at the moment, since that's what is
installed on my system. Installing this dependency is not
straight-forward on non-*nix systems so the necesarry import
statements are only executed when the relevant methods are called.

## Generic Mapping Routines

The `mapping` modules contains a number of classes that can be
subclassed to new applications. `Map()` describes a full map and will
generate a list of points on the map, called loci. Each `Locus()`
object has a `metric` attribute that contains the numerical value to
be plotted.

The generic routines have been designed with the goal of being as
extensible as possible. However, given the complex nature of these
types of experiments, extending the technique to new instruments will
require that several methods be over-ridden.

## X-Ray Diffraction

The `xrd` modules can be used to plot x-ray diffraction data. The
default implementation is meant to connect with a Bruker D8 Discover
µ-XRD using the GADDS software and a two-dimensional detector. The
`refinement` folder contains some interfaces for performing profile
refinements on diffraction data, for example FullProf. The native
refinement methods approximate some of this behavior, but with a much
reduced feature set. Care should be taken when performing automated
refinements; the fact that a refinement was successful is no guarantee
that the results are accurate.

## Full-field Transmission X-ray Microscopy

This module does not make use of the generic mapping routines, mostly
due to the image nature of the data collected. The default importer
works with data collected from the Stanford Synchrotron Light Source
(SSRL) beamline 6-2c or the Advanced Photon Source beamline 8-BM-B.

## Electrochemistry

Not related to mapping, this section contains some
importers and classes used to read in electrochemical cycling data
from several BioLogic instruments.
