.. _tabular-output-configurations:

Tabular Output Configuration
============================

Complete configuration guide for tabular prediction and classification tasks.

.. contents::
   :local:
   :depth: 2

Overview
--------

Tabular outputs handle structured predictions including:

* **Classification** - Multi-class and binary classification
* **Regression** - Continuous value prediction
* **Multi-target** - Multiple simultaneous predictions
* **Mixed targets** - Combination of classification and regression

Quick Example
-------------

.. code-block:: yaml

   output_info:
     output_source: "my_tabular_output.csv"
     output_name: "patient_outcomes"
     output_type: "tabular"
   output_type_info:
     target_cat_columns: ["diagnosis", "risk_level"]
     target_con_columns: ["survival_time", "biomarker_level"]
   model_config:
     model_type: "mlp-residual"
     model_init_config:
       fc_repr_dim: 128
       rb_do: 0.1

Output Type Configuration
-------------------------

.. autoclass:: eir.setup.schemas.TabularOutputTypeConfig

Output Module Configuration
---------------------------

Base Tabular Output Module
^^^^^^^^^^^^^^^^^^^^^^^^^^

.. autoclass:: eir.models.output.tabular.tabular_output_modules.TabularOutputModuleConfig

Available Output Architectures
-------------------------------

Residual MLP
^^^^^^^^^^^^

.. autoclass:: eir.models.output.tabular.mlp_residual.ResidualMLPOutputModuleConfig

Linear Output
^^^^^^^^^^^^^

.. autoclass:: eir.models.output.tabular.linear.LinearOutputModuleConfig