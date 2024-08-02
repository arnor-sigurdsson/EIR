.. _api-reference:

Configuration API
=================

.. contents::
   :local:
   :depth: 2


Global Configurations
---------------------

.. autoclass:: eir.setup.schemas.GlobalConfig
   :members:

Basic Experiment Configuration
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. autoclass:: eir.setup.schemas.BasicExperimentConfig
   :members:

Model Configuration
^^^^^^^^^^^^^^^^^^^

.. autoclass:: eir.setup.schemas.GlobalModelConfig
   :members:

Optimization Configuration
^^^^^^^^^^^^^^^^^^^^^^^^^^

.. autoclass:: eir.setup.schemas.OptimizationConfig
   :members:

Learning Rate Schedule Configuration
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. autoclass:: eir.setup.schemas.LRScheduleConfig
   :members:

Training Control Configuration
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. autoclass:: eir.setup.schemas.TrainingControlConfig
   :members:

Evaluation and Checkpoint Configuration
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. autoclass:: eir.setup.schemas.EvaluationCheckpointConfig
   :members:

Attribution Analysis Configuration
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. autoclass:: eir.setup.schemas.AttributionAnalysisConfig
   :members:

Metrics Configuration
^^^^^^^^^^^^^^^^^^^^^

.. autoclass:: eir.setup.schemas.SupervisedMetricsConfig
   :members:

Visualization and Logging Configuration
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. autoclass:: eir.setup.schemas.VisualizationLoggingConfig
   :members:

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

See :ref:`external-sequence-models` for more details about available
external sequence models.

.. autoclass:: eir.models.input.image.image_models.ImageModelConfig

See :ref:`external-image-models` for more details about available
external image models.

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

.. autoclass:: eir.models.input.array.models_cnn.CNNModelConfig

.. autoclass:: eir.models.input.array.models_identity.IdentityModelConfig

.. autoclass:: eir.models.input.array.models_locally_connected.SimpleLCLModelConfig

.. autoclass:: eir.models.input.array.models_locally_connected.LCLModelConfig

.. autoclass:: eir.models.input.array.models_linear.LinearModelConfig

Tabular Feature Extractors
^^^^^^^^^^^^^^^^^^^^^^^^^^

.. autoclass:: eir.models.input.tabular.tabular.SimpleTabularModelConfig

Sequence and Binary Feature Extractors
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

**Built-in Sequence Feature Extractors**

.. autoclass:: eir.models.input.sequence.transformer_models.BasicTransformerFeatureExtractorModelConfig

**External Sequence Feature Extractors**

Please refer to :ref:`external-sequence-models` for more details about
the external image models.


Image Feature Extractors
^^^^^^^^^^^^^^^^^^^^^^^^

**Built-in Image Feature Extractors**

.. autoclass:: eir.models.input.array.models_cnn.CNNModelConfig
    :noindex:

**External Image Feature Extractors**

Please refer to :ref:`external-image-models` for more details about
the external image models.

Array Feature Extractors
^^^^^^^^^^^^^^^^^^^^^^^^

.. autoclass:: eir.models.input.array.models_cnn.CNNModelConfig
    :noindex:

.. autoclass:: eir.models.input.array.models_locally_connected.LCLModelConfig
    :noindex:

.. autoclass:: eir.models.input.array.models_transformers.ArrayTransformerConfig
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

.. autoclass:: eir.setup.schema_modules.output_schemas_array.ArrayOutputTypeConfig

Output Module Configuration
^^^^^^^^^^^^^^^^^^^^^^^^^^^

**Tabular Output Modules**

.. autoclass:: eir.models.output.tabular.tabular_output_modules.TabularOutputModuleConfig

The documentation below details what the parameters passed to the respective output
output heads
of the tabular output model.
(trough the `model_init_config` field in the `\-\-output_configs` `.yaml` files).

.. autoclass:: eir.models.output.tabular.mlp_residual.ResidualMLPOutputModuleConfig

.. autoclass:: eir.models.output.tabular.linear.LinearOutputModuleConfig

**Sequence Output Modules**

.. autoclass:: eir.models.output.sequence.sequence_output_modules.SequenceOutputModuleConfig

**Array Output Modules**

.. autoclass:: eir.models.output.array.array_output_modules.ArrayOutputModuleConfig


Output Sampling Configuration
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. autoclass:: eir.setup.schema_modules.output_schemas_sequence.SequenceOutputSamplingConfig

.. autoclass:: eir.setup.schema_modules.output_schemas_array.ArrayOutputSamplingConfig


Tensor Broker Configuration
---------------------------

.. autoclass:: eir.setup.schema_modules.tensor_broker_schemas.TensorBrokerConfig

.. autoclass:: eir.setup.schema_modules.tensor_broker_schemas.TensorMessageConfig
