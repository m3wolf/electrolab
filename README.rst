SciMap: Scientific Mapping for Python
=====================================

.. image::
   https://readthedocs.org/projects/scimap/badge/?version=latest
   :target: http://scimap.readthedocs.io/en/latest/?badge=latest
   :alt: Documentation Status

Motivation
----------

This project aims to take large datasets in chemistry and physics, and
distill them down to two-dimensional spatial maps where the color
represents some physically meaningful calculated metric.

Graphical User Interface
------------------------

Several modules provide a GUI for viewing computed data. These
sections make use of **GTK3** at the moment, since that's what is
installed on my system. Installing this dependency is not
straight-forward on non-\*nix systems so the necesarry import
statements are only executed when the relevant methods are called.

Installation
------------

This project is in development and not available on pypi or conda. It
can be installed from source via pip:

.. code:: bash

   $ git clone https://github.com/m3wolf/scimap.git
   $ pip install -r scimap/requirements.txt
   $ pip install -e scimap/

Issues
------
If you have any problems or features you'd like to request, please open a new issue_:

.. _issue: https://github.com/m3wolf/scimap/issues
