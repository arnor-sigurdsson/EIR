.. _tabular-configurations:

Tabular Data Configuration
==========================

Complete configuration guide for structured tabular data (CSV).

.. contents::
   :local:
   :depth: 2

Overview
--------

Tabular data in EIR handles structured data with rows and columns, supporting both numerical and categorical features with automatic preprocessing and encoding.

Quick Example
-------------

.. code-block:: yaml

   input_info:
     input_source: "my_csv_file.csv"
     input_name: "patient_data"
     input_type: "tabular"
   input_type_info:
     input_cat_columns: ["gender", "diagnosis"]
     input_con_columns: ["age", "weight", "height"]
   model_config:
     model_type: "tabular"
     model_init_config:
       l1: 1e-04
       fc_repr_dim: 64

Input Data Configuration
------------------------

Base Configuration
^^^^^^^^^^^^^^^^^^

.. autoclass:: eir.setup.schemas.TabularInputDataConfig

Model Selection
^^^^^^^^^^^^^^^

.. autoclass:: eir.models.input.tabular.tabular.TabularModelConfig

Available Feature Extractors
-----------------------------

Simple Tabular Model
^^^^^^^^^^^^^^^^^^^^

.. autoclass:: eir.models.input.tabular.tabular.SimpleTabularModelConfig