.. _array-output-configurations:

Array Output Configuration
==========================

Complete configuration guide for multi-dimensional array and tensor output tasks.

.. contents::
   :local:
   :depth: 2

Overview
--------

Array outputs handle multi-dimensional predictions including:

* **Image generation** - Synthetic image creation
* **Signal reconstruction** - Audio, sensor data reconstruction
* **Multi-dimensional regression** - Tensor-valued predictions
* **Structured output** - Grid-based predictions

Quick Example
-------------

.. code-block:: yaml

   output_info:
     output_source: "my/array/output/folder/"
     output_name: "reconstructed_signal"
     output_type: "array"
   output_type_info:
     output_dimensions: [128, 128]
   model_config:
     model_type: "array"
     model_init_config:
       fc_repr_dim: 512

Output Type Configuration
-------------------------

.. autoclass:: eir.setup.schema_modules.output_schemas_array.ArrayOutputTypeConfig

Output Module Configuration
---------------------------

.. autoclass:: eir.models.output.array.array_output_modules.ArrayOutputModuleConfig

Output Sampling Configuration
-----------------------------

.. autoclass:: eir.setup.schema_modules.output_schemas_array.ArrayOutputSamplingConfig