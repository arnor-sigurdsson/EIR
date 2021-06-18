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
    --n_epochs 20 \
    --dataloader_workers 0 \
    --omics_sources processed_sample_data/arrays/ \
    --omics_names genotype \
    --sample_interval 100 \
    --checkpoint_interval 100 \
    --n_saved_models 1 \
    --snp_file processed_sample_data/data_final_gen.bim  \
    --label_file processed_sample_data/human_origins_labels.csv  \
    --target_cat_columns Origin  \
    --model_type genome-local-net \
    --fc_task_dim 32 `#Number of hidden nodes in fully-connected layers` \
    --fc_do 0.5 `#Last layer dropout` \
    --rb_do 0.5 `#Residual blocks dropout` \
    --na_augment_perc 0.4 `#Input NA dropout` \
    --na_augment_prob 1.0 `#Probability of applying input NA dropout` \
    --layers 2 \
    --run_name gln \
    --channel_exp_base 1 `#Use 2**1 weight sets in locally-connected layers` \
    --kernel_width 8 `#Locally connected with, 8 / 2 = 2 SNPs` \
    --get_acts \
    --max_acts_per_class 200 `#Limit number of samples used for activation computations` \


.. tip::

    For a full set of options and a short explanation of each, run ``eirtrain --help``.


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


Predicting on external samples
------------------------------

To predict on unseen samples,
we can use the ``eirpredict``, for example like so:

.. code-block:: console

    eirpredict --model_path runs/gln/saved_models/<chosen_model>  --output_folder runs --device cpu --label_file <path/to/test_labels.csv>  --omics_sources <path/to/test_arrays> --omics_names genotype --evaluate

This will generate a folder called ``test_set_predictions``
under in the ``--output_folder`` directory.

.. tip::

   We might want to predict on samples where we do not have any labels,
   in that case just omit the ``--evaluate`` and ``--label_file`` flags.

Applying to other datasets
--------------------------

Hopefully this small demo was useful!
To apply to your own data,
you will have to process it (see: `plink pipelines`_)
and change the following flags:

.. code-block:: console

    --omics_sources <path/to/your/processed/arrays> --snp_file <path/to/your/bim/file> --label_file <path/to/your/labels/csv/file> --target_cat_columns <name_of_target_column_in_label_file>

