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
(trough the `model_config` field in the `\-\-input_configs` `.yaml` files).


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

.. autoclass:: eir.models.sequence.transformer_models.BasicTransformerFeatureExtractorModelConfig

Predictor Configurations
------------------------

Note that the settings here currently only
refer to the predictor branches.
Future work includes
adding more configurations
to the fusion operations themselves,
and having therefore separate configurations
for those and the predictor branches.

.. autoclass:: eir.setup.schemas.PredictorConfig

Predictor Modules
^^^^^^^^^^^^^^^^^

.. autoclass:: eir.models.fusion.fusion_default.FusionModelConfig

.. autoclass:: eir.models.fusion.fusion_mgmoe.MGMoEModelConfig

.. autoclass:: eir.models.fusion.fusion_linear.LinearFusionModelConfig

Target Configurations
---------------------

.. autoclass:: eir.setup.schemas.TargetConfig