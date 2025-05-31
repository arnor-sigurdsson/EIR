.. _image-configurations:

Image Data Configuration
========================

Complete configuration guide for image and visual data processing.

.. contents::
   :local:
   :depth: 2


Quick Example
-------------

.. code-block:: yaml

   input_info:
     input_source: "my/image/folder/"
     input_name: "chest_xray"
     input_type: "image"
   input_type_info:
     size: [224, 224]
     num_channels: 1
   model_config:
     model_type: "cnn"
     model_init_config:
       channel_exp_base: 4
       down_stride_average: True
       kernel_width: 3

Input Data Configuration
------------------------

Base Configuration
^^^^^^^^^^^^^^^^^^

.. autoclass:: eir.setup.schemas.ImageInputDataConfig

Model Selection
^^^^^^^^^^^^^^^

.. autoclass:: eir.models.input.image.image_models.ImageModelConfig

Available Feature Extractors
-----------------------------

Built-in Image Models
^^^^^^^^^^^^^^^^^^^^^

CNN Architecture
""""""""""""""""

.. autoclass:: eir.models.input.array.models_cnn.CNNModelConfig

External Image Models
^^^^^^^^^^^^^^^^^^^^^

For pre-trained vision models (ResNet, Vision Transformers, etc.), please refer to :ref:`external-image-models` for detailed configuration options.

Interpretation Support
----------------------

.. autoclass:: eir.setup.schemas.BasicInterpretationConfig