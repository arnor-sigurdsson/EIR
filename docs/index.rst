EIR
===

``EIR`` is a framework for training linear and
deep learning models on genotype and tabular data.

Features
--------

- **Adaptive**: Deep models that automatically adapt
  to genotype size. No changes needed
  whether training on 1,000 or 803,113 SNPs.
- **Integration**: Integrate tabular and
  genotype data. Include e.g. biochemical
  measurements for your modelling.
- **Multi-task learning**: Train on multiple
  categorical/continuous targets at the same time.
- **Customizable**: Through the Python interface, use your
  own models and metrics.

Installation
------------

.. code-block:: console

    $ pip install eir-dl


Documentation
-------------

To get started, please, read :doc:`quickstart`

.. toctree::
    :maxdepth: 2

    quickstart
    tutorials/tutorial_index
    binaries_guide
    api_reference
    license