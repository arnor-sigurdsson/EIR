.. _global-configurations:

Global Configurations
=====================

Core configuration classes for experiment setup, training, and optimization.
The root object is the :class:`eir.setup.schemas.GlobalConfig`,
which contains all other configurations split by "theme" / functionality.
While there are a lot of options available, one can start with a minimal configuration
such as the following:

Quick Example
-------------

.. code-block:: yaml

    basic_experiment:
      output_folder: my/experiment/folder
    evaluation_checkpoint:
      checkpoint_interval: 200
      sample_interval: 200

.. contents::
   :local:
   :depth: 2

Below is a detailed overview of all the global configuration options available in EIR.

.. autoclass:: eir.setup.schemas.GlobalConfig
   :members:

Basic Setup
-----------

.. autoclass:: eir.setup.schemas.BasicExperimentConfig
   :members:

Model and Training
------------------

.. autoclass:: eir.setup.schemas.GlobalModelConfig
   :members:

.. autoclass:: eir.setup.schemas.OptimizationConfig
   :members:

.. autoclass:: eir.setup.schemas.LRScheduleConfig
   :members:

Training Control
----------------

.. autoclass:: eir.setup.schemas.TrainingControlConfig
   :members:

.. autoclass:: eir.setup.schemas.EvaluationCheckpointConfig
   :members:

Analysis and Logging
--------------------

.. autoclass:: eir.setup.schemas.AttributionAnalysisConfig
   :members:

.. autoclass:: eir.setup.schemas.SupervisedMetricsConfig
   :members:

.. autoclass:: eir.setup.schemas.VisualizationLoggingConfig
   :members:

Infrastructure
--------------

.. autoclass:: eir.setup.schemas.DataPreparationConfig
   :members:

.. autoclass:: eir.setup.schemas.AcceleratorConfig
   :members: