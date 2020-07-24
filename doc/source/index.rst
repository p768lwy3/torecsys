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

* Numpy >= 1.17.0
* Pandas >= 0.24.2
* PyTorch >= 1.2

Installation
============

Install with pip:

.. code-block:: console

    pip install torecsys

Install with source code:

.. code-block:: console

    git clone https://github.com/p768lwy3/torecsys.git
    cd ./torecsys
    python setup.py build
    python setup.py install

Notations in documentation
==========================
.. list-table:: notations
    :widths: 50 50
    :header-rows: 1

    * - Notation
      - Refer to
    * - **T**
      - torch.Tensor
    * - **B**
      - batch size
    * - **E**
      - embedding size
    * - **H_i**
      - output size of i-th hidden layer
    * - **N**
      - number of fields
    * - **I**
      - input sizes of any layers required *inputs_size*
    * - **O**
      - output size of any layers requried *output_size*
    * - **V**
      - total number of words in a vocabulary set
    * - **S**
      - total number of samples, e.g. negative samples


API documentation
=================

.. toctree::
   :maxdepth: 1
   :caption: Package Reference
   
   modules.rst
   torecsys.data.rst
   torecsys.functional.rst
   torecsys.inputs.rst
   torecsys.layers.rst
   torecsys.inputs.rst
   torecsys.losses.rst
   torecsys.metrics.rst
   torecsys.models.rst
   torecsys.utils.rst


Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
