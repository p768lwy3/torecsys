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

Notations
=========
+---------+--------------------------------------------------+
| Symbol: | To refer:                                        |
+=========+==================================================+
| **T**   | torch.Tensor                                     |
| **B**   | batch size                                       |
| **E**   | embedding size                                   |
| **H_i** | output size of i-th hidden layer                 |
| **N**   | number of fields                                 |
| **I**   | input sizes of any layers required *inputs_size* |
| **O**   | output size of any layers requried *output_size* |
| **V**   | total number of words in a vocabulary set        |
| **S**   | total number of samples, e.g. negative samples   |
+---------+--------------------------------------------------+


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
