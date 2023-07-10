.. _api-reference:

API
===

.. contents::
   :local:
   :depth: 2


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

.. autoclass:: eir.setup.schemas.ArrayInputDataConfig

Input Model Configurations
^^^^^^^^^^^^^^^^^^^^^^^^^^

These configurations are used to specify the
input feature extractor architecture, as
well as paramters that can be common between different feature extractors.
For a given feature extractor (specified with the `model_type` field), there
are there are various configurations available through the `model_init_config`
field. The documentation below contains more details about the different
configurations available for each feature extractor.

.. autoclass:: eir.models.input.omics.omics_models.OmicsModelConfig

.. autoclass:: eir.models.input.tabular.tabular.TabularModelConfig

.. autoclass:: eir.models.input.sequence.transformer_models.SequenceModelConfig

.. autoclass:: eir.models.input.image.image_models.ImageModelConfig

.. autoclass:: eir.models.input.array.array_models.ArrayModelConfig


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

.. autoclass:: eir.models.input.omics.models_cnn.CNNModelConfig

.. autoclass:: eir.models.input.omics.models_identity.IdentityModelConfig

.. autoclass:: eir.models.input.omics.models_locally_connected.SimpleLCLModelConfig

.. autoclass:: eir.models.input.omics.models_locally_connected.LCLModelConfig

.. autoclass:: eir.models.input.omics.models_linear.LinearModelConfig

Tabular Feature Extractors
^^^^^^^^^^^^^^^^^^^^^^^^^^

.. autoclass:: eir.models.input.tabular.tabular.SimpleTabularModelConfig

Sequence and Binary Feature Extractors
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. autoclass:: eir.models.input.sequence.transformer_models.BasicTransformerFeatureExtractorModelConfig

Image Feature Extractors
^^^^^^^^^^^^^^^^^^^^^^^^

Please refer to `this page <https://huggingface.co/docs/timm/models>`_
for information about the image models.


Array Feature Extractors
^^^^^^^^^^^^^^^^^^^^^^^^

.. autoclass:: eir.models.input.omics.models_cnn.CNNModelConfig
    :noindex:

.. autoclass:: eir.models.input.omics.models_locally_connected.LCLModelConfig
    :noindex:


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

.. autoclass:: eir.setup.schema_modules.output_schemas_sequence.SequenceOutputTypeConfig

Output Module Configuration
^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. autoclass:: eir.models.output.output_module_setup.TabularOutputModuleConfig

.. autoclass:: eir.models.output.sequence.sequence_output_modules.SequenceOutputModuleConfig

The documentation below details what the parameters passed to the respective output
output heads
(trough the `model_init_config` field in the `\-\-output_configs` `.yaml` files).

.. autoclass:: eir.models.output.mlp_residual.ResidualMLPOutputModuleConfig

.. autoclass:: eir.models.output.linear.LinearOutputModuleConfig

Output Sampling Configuration
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. autoclass:: eir.setup.schema_modules.output_schemas_sequence.SequenceOutputSamplingConfig

