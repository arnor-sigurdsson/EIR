.. _image-output-configurations:

Image Output Configuration
==========================

Complete configuration guide for image generation and reconstruction tasks.

.. contents::
   :local:
   :depth: 2

Overview
--------

Image outputs handle visual data generation including:

* **Image synthesis** - Creating new images from learned distributions
* **Image reconstruction** - Restoring degraded or incomplete images
* **Style transfer** - Converting images between different styles
* **Medical image generation** - Synthetic medical imaging data

Quick Example
-------------

.. code-block:: yaml

   output_info:
     output_source: "my//images/"
     output_name: "synthetic_xray"
     output_type: "image"
   output_type_info:
     output_dimensions: [256, 256]
     num_channels: 1
   model_config:
     model_type: "array"  # Images use array-based architectures
     model_init_config:
       fc_repr_dim: 1024

Output Type Configuration
-------------------------

.. autoclass:: eir.setup.schema_modules.output_schemas_image.ImageOutputTypeConfig

Output Module Configuration
---------------------------

Image outputs typically use array-based output modules. See :doc:`array_output` for detailed configuration options.