.. _survival-output-configurations:

Survival Analysis Output Configuration
======================================

Complete configuration guide for survival analysis and time-to-event modeling.

.. contents::
   :local:
   :depth: 2

Overview
--------

Survival outputs handle time-to-event analysis including:

* **Survival prediction** - Time until event occurrence
* **Hazard modeling** - Risk assessment over time
* **Censored data handling** - Incomplete observation modeling
* **Medical prognosis** - Patient outcome prediction

Quick Example
-------------

.. code-block:: yaml

   output_info:
     output_name: "patient_survival"
     output_type: "survival"
   output_type_info:
     time_column: "survival_days"
     event_column: "death_observed"
   model_config:
     model_type: "mlp-residual"
     model_init_config:
       fc_repr_dim: 64

Output Type Configuration
-------------------------

.. autoclass:: eir.setup.schema_modules.output_schemas_survival.SurvivalOutputTypeConfig

Output Module Configuration
---------------------------

Survival analysis typically uses tabular-based output modules. See :doc:`tabular_output` for detailed configuration options of the underlying architectures.