.. _api-reference:

API
===


Global Configurations
---------------------

.. autoclass:: eir.setup.schemas.GlobalConfig

Input Configurations
--------------------

.. autoclass:: eir.setup.schemas.InputConfig

Input Data Configuration
^^^^^^^^^^^^^^^^^^^^^^^^

.. autoclass:: eir.setup.schemas.InputDataConfig

Input Type Configurations
^^^^^^^^^^^^^^^^^^^^^^^^^

.. autoclass:: eir.setup.schemas.OmicsInputDataConfig

.. autoclass:: eir.setup.schemas.TabularInputDataConfig

.. autoclass:: eir.setup.schemas.SequenceInputDataConfig

.. autoclass:: eir.setup.schemas.ByteInputDataConfig

.. autoclass:: eir.setup.schemas.ImageInputDataConfig

Interpretation Configurations
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Parameters to have basic control over how interpretation is done. Currently only
supported for sequence and image data.

.. autoclass:: eir.setup.schemas.BasicInterpretationConfig

Feature Extractor Configurations
--------------------------------

The documentation below details what the parameters passed to the respective models
(trough the `model_init_config` field in the `\-\-input_configs` `.yaml` files).


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

Sequence Feature Extractors
^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. autoclass:: eir.models.sequence.transformer_models.SequenceModelConfig

.. autoclass:: eir.models.sequence.transformer_models.BasicTransformerFeatureExtractorModelConfig

Image Feature Extractors
^^^^^^^^^^^^^^^^^^^^^^^^

.. autoclass:: eir.models.image.image_models.ImageModelConfig

Fusion Configurations
---------------------

.. autoclass:: eir.setup.schemas.FusionConfig

Fusion Module Configuration
^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. autoclass:: eir.models.fusion.fusion_default.ResidualMLPConfig

.. autoclass:: eir.models.fusion.fusion_mgmoe.MGMoEModelConfig

.. autoclass:: eir.models.fusion.fusion_identity.IdentityConfig

Output Configurations
---------------------

.. autoclass:: eir.setup.schemas.OutputConfig

Output Info Configuration
^^^^^^^^^^^^^^^^^^^^^^^^^

.. autoclass:: eir.setup.schemas.OutputInfoConfig

Output Type Configuration
^^^^^^^^^^^^^^^^^^^^^^^^^

.. autoclass:: eir.setup.schemas.TabularOutputTypeConfig

Output Module Configuration
^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. autoclass:: eir.models.output.tabular_output.TabularModelOutputConfig

The documentation below details what the parameters passed to the respective output
output heads
(trough the `model_init_config` field in the `\-\-output_configs` `.yaml` files).

.. autoclass:: eir.models.output.tabular_output.TabularMLPResidualModelConfig
