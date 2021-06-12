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

To train a MLP model on this data, which has 6 ancestry target classes, computing SNP activations during training, we can run the following command:

```bash
eirtrain \
--n_epochs 20 \
--omics_sources processed_sample_data/arrays/ \
--omics_names genotype \
--sample_interval 200 \
--snp_file processed_sample_data/data_final_gen.bim  \
--label_file processed_sample_data/human_origins_labels.csv  \
--lr 1e-3 \
--lr_schedule cosine \
--lr_lb 0.0 \
--batch_size 32  \
--target_cat_columns Origin  \
--model_type mlp \
--fc_repr_dim 6 \
--fc_task_dim 64 \
--optimizer adam \
--warmup_steps auto \
--plot_skip_steps 0 \
--wd 1e-3 \
--l1 1e-4 \
--fc_do 0.5 \
--rb_do 0.5 \
--na_augment_perc 0.20 \
--na_augment_prob 1.0 \
--layers 2 \
--run_name test_mlp \
--dataloader_workers 0 \
--get_acts
```

