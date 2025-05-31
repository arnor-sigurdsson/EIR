.. _omics-configurations:

Omics Data Configuration
========================

Configuration guide for genomics input data in EIR.

.. contents::
   :local:
   :depth: 2

Overview
--------

Omics data in EIR requires two main configuration components:

1. **Input Data Configuration** - defines data source and preprocessing
2. **Feature Extractor Configuration** - defines the model architecture

Quick Example
-------------

.. code-block:: yaml

   input_info:
     input_source: "my/folder/path"
     input_name: "my_omics"
     input_type: "omics"
   input_type_info:
     snp_file: "by_bim.bim"
   model_config:
     model_type: "cnn"
     model_init_config:
       channel_exp_base: 3
       kernel_width: 8

Input Data Configuration
------------------------

Base Configuration
^^^^^^^^^^^^^^^^^^

.. autoclass:: eir.setup.schemas.OmicsInputDataConfig

Model Selection
^^^^^^^^^^^^^^^

.. autoclass:: eir.models.input.omics.omics_models.OmicsModelConfig

Available Feature Extractors
-----------------------------

CNN Models
^^^^^^^^^^

.. autoclass:: eir.models.input.array.models_cnn.CNNModelConfig

Linear Models
^^^^^^^^^^^^^

.. autoclass:: eir.models.input.array.models_linear.LinearModelConfig

Locally Connected Models
^^^^^^^^^^^^^^^^^^^^^^^^

.. autoclass:: eir.models.input.array.models_locally_connected.SimpleLCLModelConfig

.. autoclass:: eir.models.input.array.models_locally_connected.LCLModelConfig

Identity Models
^^^^^^^^^^^^^^^

.. autoclass:: eir.models.input.array.models_identity.IdentityModelConfig