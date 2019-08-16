.. torecsys documentation master file, created by
   sphinx-quickstart on Wed Aug 14 16:39:23 2019.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

:github_url: https://github.com/p768lwy3/torecsys

Welcome to torecsys's documentation!
====================================

*Recommendation System in PyTorch.*

This package is an implementation of several famous recommendation systelm algorithm 
in PyTorch, including click-through-rate prediction, learning-to-ranking and embedding. 
ToR[e]csys also comes with data loader, layers-level implementation, self-defined 
architecture of models. It's open-source software, released under the MIT license.


Minimial Requirements
=====================

* (on planning...)

Installation
============

Install with pip:

    pip install torecsys (planning to develop on 0.1.0...)

API documentation
=================

.. toctree::
    :maxdepth: 1
    :caption: Package Reference
    source/torecsys.data
    source/torecsys.functional
    source/torecsys.inputs
    source/torecsys.layers
    source/torecsys.inputs
    source/torecsys.losses
    source/torecsys.metrics
    source/torecsys.models
    source/torecsys.utils


Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
