EIR Tutorial: Genetic and Multimodal Survival Prediction
========================================================

This tutorial demonstrates how to use EIR for survival prediction using genetic data, both alone and in combination with clinical variables. We'll cover two scenarios:

1. Training a survival model using only genetic data
2. Training a multimodal model using both genetic and clinical data

For a more detailed introduction to EIR's genetic prediction capabilities, see the `Genotype Tutorial <https://eir.readthedocs.io/en/stable/tutorials/a_using_eir/01_basic_tutorial.html>`_. For more details about survival analysis in EIR, see the `Survival Analysis Tutorial <https://eir.readthedocs.io/en/stable/tutorials/h_survival_analysis/02_survival_flchain_cox.html>`_.

Project Setup
-------------

First, create a directory structure for your project::

    project_directory/
    ├── conf/
    │   ├── global_config.yaml
    │   ├── input_genotype_config.yaml
    │   ├── input_tabular_config.yaml
    │   ├── fusion_config.yaml
    │   └── output_config.yaml
    └── data/
        ├── arrays/          # Your genetic data arrays
        ├── genotype.bim     # SNP information file
        └── phenotype.csv    # Clinical and survival data

To prepare the array folder, you can use the `plink-pipelines <https://github.com/arnor-sigurdsson/plink_pipelines>`_ software::

    plink_pipelines --raw_data_path <folder with plink bed/fam/bim fileset> --output_folder data/arrays

Configuration Files
-------------------

Below are only shown parts of each configuration file, please refer to the full configuration files in the supplementary data for full configurations.

Global Configuration (global_config.yaml)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: yaml

    basic_experiment:
      n_epochs: 5000
      output_folder: FILL # Path where you want to save your experiment
      batch_size: 64
      valid_size: 2000

    lr_schedule:
      lr_schedule: plateau
      lr_plateau_factor: 0.2
      lr_plateau_patience: 6

    optimization:
      optimizer: adabelief
      lr: 5.0e-05
      wd: 0.0001

[rest of configs follow same pattern...]

Training Models
---------------

Genetics-Only Model
~~~~~~~~~~~~~~~~~~~

To train a model using only genetic data:

.. code-block:: bash

    eirtrain \
    --global_configs conf/global_config.yaml \
    --input_configs conf/input_genotype_config.yaml \
    --fusion_configs conf/fusion_config.yaml \
    --output_configs conf/output_config.yaml

Multimodal Model (Genetics + Clinical)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

To train a model using both genetic and clinical data:

.. code-block:: bash

    eirtrain \
    --global_configs conf/global_config.yaml \
    --input_configs conf/input_genotype_config.yaml conf/input_tabular_config.yaml \
    --fusion_configs conf/fusion_config.yaml \
    --output_configs conf/output_config.yaml

Model Evaluation
----------------

The training will generate several outputs in your specified output folder:

- Training curves showing loss and performance metrics
- Survival curves for validation samples
- Feature importance analysis for both genetic and clinical variables (if used, see ``compute_attributions`` in ``global_config.yaml`` supplementary file)
- Saved model checkpoints

To evaluate a trained model on new data:

.. code-block:: bash

    eirpredict \
    --global_configs conf/global_config.yaml \
    --input_configs conf/input_genotype_config.yaml \  # Add tabular config for multimodal
    --output_configs conf/output_config.yaml \
    --model_path path/to/saved/model.pt \
    --evaluate \
    --output_folder path/to/prediction/output

Note that for evaluation, relevant filepaths in the configurations must be updated to point to the test data / data to predict on.

Notes
-----
- Make sure to replace all ``FILL`` placeholders with appropriate paths and values

For more detailed information about configuration options and advanced features, refer to the `EIR documentation <https://eir.readthedocs.io/>`_.