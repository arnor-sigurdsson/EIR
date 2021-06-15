<p align="center">
  <img src="docs/source/_static/img/EIR_logo.png">
</p>

<p align="center">
    <a href="LICENSE" alt="License">
        <img src="https://img.shields.io/badge/License-APGL-5B2D5B.svg" /></a>
  
  <a href="https://www.biorxiv.org/content/10.1101/2021.06.11.447883v1" alt="bioRxiv">
        <img src="https://img.shields.io/badge/Paper-bioRxiv-B5232F.svg" /></a>
   
</p>

---

Supervised modelling on genotype and tabular data.

**WARNING:** This project is in alpha phase. Expect backwards incompatiable changes and API changes.

## Install

`pip install eir-dl`

## Quickstart

Download [sample data](https://drive.google.com/file/d/17vzG8AXVD684HqTD6RNtKjrK8tzHWeGx/view?usp=sharing) (or process your own `.bed`, `.bim`, `.fam` files with e.g. [plink_pipelines](https://github.com/arnor-sigurdsson/plink_pipelines)).

Examining the sample data, we can see the following structure:

```bash
processed_sample_data
├── arrays                      # Genotype data as NumPy arrays
├── data_final_gen.bim          # Variant information file accompanying the genotype arrays
└── human_origins_labels.csv    # Contains the target labels (what we want to predict from the genotype data)
```
> **_NOTE:_**  Currently the label file ID columns must be called "ID" (uppercase).

To train a GLN model on this data, which has 6 ancestry target classes, computing SNP activations during training, we can run the following command:

```bash
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
--fc_task_dim 32 \
--fc_do 0.5 \
--rb_do 0.5 \
--na_augment_perc 0.4 \
--na_augment_prob 1.0 \
--layers 2 \
--run_name gln \
--channel_exp_base 1 \
--kernel_width 8 \
--memory_dataset
```

> **_TIP:_**  For a full set of options and a short explanation of each, run `eirtrain --help`.

This will generate a folder called `runs/test_mlp` in the current directory containing the results of the training. The folder has roughly the following structure (some files/folders are omitted here):

```bash
├── cl_args.json
├── model_info.txt
├── saved_models
├── results
│   └── Origin  # Target column
│       ├── samples
│       │   ├── 100 # Validation results according to --sample_interval
│       │   │   ├── activations # Activations, computed if --get_acts flag is used
│       │   │   ├── confusion_matrix.png
│       │   │   ├── mc_pr_curve.png
│       │   │   ├── mc_roc_curve.png
│       │   │   └── wrong_preds.csv
│       │   ├── 200
│       │   │   ├── ...
│       │   ├── 300
│       │   │   ├── ...
│       ├── training_curve_ACC.png
│       ├── training_curve_AP-MACRO.png
│       ├── training_curve_LOSS.png
│       ├── training_curve_MCC.png
│       ├── training_curve_ROC-AUC-MACRO.png
├── training_curve_LOSS-AVERAGE.png
├── training_curve_PERF-AVERAGE.png
```

### Predicting on external samples

To predict on unseen samples, we can use the `eirpredict`, for example like so:

```bash
eirpredict --model_path runs/gln/saved_models/<chosen_model>  --output_folder runs --device cpu --label_file <path/to/test_labels.csv>  --omics_sources <path/to/test_arrays> --omics_names genotype --evaluate
```

This will generate a folder called `test_set_predictions` under in the `--output_folder` directory.

> **_TIP:_** We might want to predict on samples where we do not have any labels, in that case just omit the `--evaluate` and `--label_file` flags.

## Citation

```
@article{sigurdsson2021deep,
  title={Deep integrative models for large-scale human genomics},
  author={Sigurdsson, Arnor Ingi and Westergaard, David and Winther, Ole and Lund, Ole and Brunak, S{\o}ren and Vilhjalmsson, Bjarni J and Rasmussen, Simon},
  journal={bioRxiv},
  year={2021},
  publisher={Cold Spring Harbor Laboratory}
}
