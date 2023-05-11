EIR
===

``EIR`` is a framework
for supervised training
of deep learning models
on genotype, tabular, sequence, image, array and binary data.
It is designed to provide
a high-level, yet modular API
that reduces the amount of boilerplate code
and pre-processing required to train a model.

.. figure:: source/_static/img/EIR_data_supported.png
   :width: 85%
   :align: center

|
Installation
------------

.. code-block:: console

    $ pip install eir-dl

.. important::
    The latest version of EIR supports
    `Python 3.11 <https://www.python.org/downloads/>`_.
    Using an older version of Python will install an old version of EIR,
    which is likely to be incompatible with the current documentation
    and may contain bugs.
    Please make sure that you are installing EIR in a Python 3.11 environment.


Documentation
-------------

To get started, please read :ref:`01-genotype-tutorial`.

.. toctree::
    :maxdepth: 2

    tutorials/tutorial_index
    api_reference
    license
    acknowledgements