.. _array-configurations:

Array Data Configuration
========================

Configuration guide for multi-dimensional array and tensor data.
Generally refers to NumPy arrays stored on disk (or streamed if using
the streaming functionality),
all of them with the same shape and type.

.. contents::
   :local:
   :depth: 2

Overview
--------

Array data in EIR handles multi-dimensional numerical data:

* **Scientific data** - Sensor readings, measurements
* **Signal processing** - Audio spectrograms, time-frequency data
* **Multi-dimensional features** - Engineered feature matrices
* **Tensor data** - Any N-dimensional numerical array

Quick Example
-------------

.. code-block:: yaml

   input_info:
     input_source: "my/array/data/folder/"
     input_name: "sensor_data"
     input_type: "array"
   model_config:
     model_type: "cnn"
     model_init_config:
       channel_exp_base: 3
       kernel_width: 5
       kernel_height: 1

Input Data Configuration
------------------------

Base Configuration
^^^^^^^^^^^^^^^^^^

.. autoclass:: eir.setup.schemas.ArrayInputDataConfig

Model Selection
^^^^^^^^^^^^^^^

.. autoclass:: eir.models.input.array.array_models.ArrayModelConfig

Available Feature Extractors
-----------------------------

CNN Models
^^^^^^^^^^

.. autoclass:: eir.models.input.array.models_cnn.CNNModelConfig

Locally Connected Models
^^^^^^^^^^^^^^^^^^^^^^^^

.. autoclass:: eir.models.input.array.models_locally_connected.LCLModelConfig

Transformer Models
^^^^^^^^^^^^^^^^^^

.. autoclass:: eir.models.input.array.models_transformers.ArrayTransformerConfig