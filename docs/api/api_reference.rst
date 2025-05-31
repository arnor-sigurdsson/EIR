.. _api-reference:

Configuration API Reference
===========================

**Searching for specific configurations?** Use the search bar at the top-left of the page.

Configuration guide organized by theme.

.. contents::
   :local:
   :depth: 1

Global Settings
---------------

Training, optimization, hardware setup and more.

.. toctree::
   :maxdepth: 2

   global_configs

Input Types
-----------

Configure how your data is processed and which models extract features from it.

.. toctree::
   :maxdepth: 1

   input_configs/omics_input
   input_configs/tabular_input
   input_configs/sequence_input
   input_configs/image_input
   input_configs/array_input
   input_configs/byte_input

Fusion
------

Combining multiple data types.

.. toctree::
   :maxdepth: 2

   fusion_configs

Output Types
------------

Configure what your model predicts or generates.

.. toctree::
   :maxdepth: 1

   output_configs/tabular_output
   output_configs/sequence_output
   output_configs/array_output
   output_configs/image_output
   output_configs/survival_output

Advanced
--------

Advanced operations.

.. toctree::
   :maxdepth: 2

   tensor_broker_configs