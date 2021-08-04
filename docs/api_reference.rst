API
===


Experiment Configuration
------------------------

.. automodule:: eir.setup.schemas
    :members:

Model Configurations
--------------------

The documentation below details what the parameters passed to the respective models
(trough the `model_config` field in the `\-\-input_configs` `.yaml` files).

Fusion Modules
^^^^^^^^^^^^^^

.. autoclass:: eir.models.fusion_linear.LinearFusionModelConfig

.. autoclass:: eir.models.fusion.FusionModelConfig

.. autoclass:: eir.models.fusion_mgmoe.MGMoEModelConfig

Omics Feature Extractors
^^^^^^^^^^^^^^^^^^^^^^^^

.. autoclass:: eir.models.omics.models_cnn.CNNModelConfig

.. autoclass:: eir.models.omics.models_identity.IdentityModelConfig

.. autoclass:: eir.models.omics.models_locally_connected.SimpleLCLModelConfig

.. autoclass:: eir.models.omics.models_locally_connected.LCLModelConfig

.. autoclass:: eir.models.omics.models_linear.LinearModelConfig

Tabular Feature Extractors
^^^^^^^^^^^^^^^^^^^^^^^^^^

.. autoclass:: eir.models.tabular.tabular.TabularModelConfig
