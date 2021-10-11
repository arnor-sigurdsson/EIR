.. _quickstart:

Quickstart
==========

Download `processed sample data`_
(or process your own `.bed`, `.bim`, `.fam` files
with e.g. `plink pipelines`_).
The sample data we are using here is for predicting ancestry
in the public `Human Origins`_ dataset,
but the same approach can just as well be used for
e.g. disease predictions in other cohorts
(for example the `UK Biobank`_).

.. _processed sample data: https://drive.google.com/file/d/17vzG8AXVD684HqTD6RNtKjrK8tzHWeGx/view?usp=sharing
.. _plink pipelines: https://github.com/arnor-sigurdsson/plink_pipelines
.. _Human Origins: https://www.nature.com/articles/nature13673
.. _UK Biobank: https://www.nature.com/articles/s41586-018-0579-z

Examining the sample data, we can see the following structure:

.. code-block:: console

    processed_sample_data
    ├── arrays                      # Genotype data as NumPy arrays
    ├── data_final_gen.bim          # Variant information file accompanying the genotype arrays
    └── human_origins_labels.csv    # Contains the target labels (what we want to predict from the genotype data)

.. important::

    Currently the label file ID columns must be called "ID" (uppercase).

For this quick demo,
we are going to use the data above to train a GLN model
to predict ancestry, of which there are 6 classes
(Asia, Eastern Asia, Europe, Latin America and the Caribbean, Middle East and Sub-Saharan Africa).
To train the model, computing SNP activations during training,
we can run the following command:

.. code-block:: bash

    eirtrain \
    --preset gln --gln_targets.label_file="processed_sample_data/human_origins_labels.csv" \
    --gln_targets.target_cat_columns="['Origin',]" \
    --gln_input.input_info.input_source="processed_sample_data/arrays/" \
    --gln_input.input_type_info.snp_file="processed_sample_data/data_final_gen.bim"

.. tip::

    While concise,
    the command above indeed obscures a lot of the configuration and functionality
    happening behind the scenes. See the :ref:`01-basic-tutorial`
    for a more thorough example.

This will generate a folder called ``runs/gln``
in the current directory containing the results of the training.
The folder has roughly the following structure
(some files/folders are omitted here):

.. code-block:: console

    ├── cl_args.json
    ├── model_info.txt
    ├── saved_models
    ├── results
    │   └── Origin  # Target column
    │       ├── samples
    │       │   ├── 100 # Validation results according to --sample_interval
    │       │   │   ├── activations # Activations, computed if --get_acts flag is used
    │       │   │   ├── confusion_matrix.png
    │       │   │   ├── mc_pr_curve.png
    │       │   │   ├── mc_roc_curve.png
    │       │   │   └── wrong_preds.csv
    │       │   ├── 200
    │       │   │   ├── ...
    │       │   ├── 300
    │       │   │   ├── ...
    │       ├── training_curve_ACC.png
    │       ├── training_curve_AP-MACRO.png
    │       ├── training_curve_LOSS.png
    │       ├── training_curve_MCC.png
    │       ├── training_curve_ROC-AUC-MACRO.png
    ├── training_curve_LOSS-AVERAGE.png
    ├── training_curve_PERF-AVERAGE.png

Hopefully this small demo was useful! For a more thorough tutorial
(e.g. showing how you can predict on external samples,
tips on applying the framework to your own data),
head to :ref:`01-basic-tutorial`.