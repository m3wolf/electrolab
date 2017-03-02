==============
 Introduction
==============

Motivation
==========

Sorry, all out of motivation for today. Check back tomorrow.

Installation
============

The easiest way to run the development version is to use pip's
**developer mode** (-e). First download the repository then install using
pip. Be aware that downloading the repository may take a while as the
test data can be very large.

.. code:: bash

   $ git clone https://github.com/m3wolf/scimap.git
   $ pip install -r scimap/requirements.txt
   $ pip install -e scimap/

Now you should be able to import scimap in your python interpreter.

.. code:: python

>>> import scimap

Running Tests
-------------

There is a set of unit tests and example data that accompany this
project. To run the tests, install the project as described above then
execute the test runner:

.. code:: bash

   $ python scimap/tests/tests.py

and you should see the following output::
  
   ...........x......x..x.xxx....xxxx.xxx.xxx......xx....x.........
   ----------------------------------------------------------------------
   Ran 64 tests in 31.432s

   OK (expected failures=19)


