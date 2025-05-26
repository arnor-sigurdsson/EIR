<p align="center">
  <img src="docs/source/_static/img/EIR_logo.svg", width="500">
</p>

<p align="center">
  <a href="LICENSE" alt="License">
        <img src="https://img.shields.io/badge/License-Apache_2.0-5B2D5B.svg" /></a>
  
  <a href="https://github.com/arnor-sigurdsson/EIR#citation" alt="Citation">
        <img src="https://img.shields.io/badge/Papers-View%20Here-5F9EA0.svg" /></a>
  
  <a href="https://www.python.org/downloads/" alt="Python">
        <img src="https://img.shields.io/badge/Python-3.13-blue.svg" /></a>
  
  <a href="https://pypi.org/project/eir-dl/" alt="Python">
        <img src="https://img.shields.io/pypi/v/eir-dl.svg" /></a>
  
  <a href="https://codecov.io/gh/arnor-sigurdsson/EIR" alt="Coverage">
        <img src="https://codecov.io/gh/arnor-sigurdsson/EIR/branch/master/graph/badge.svg" /></a>
  
  <a href='https://eir.readthedocs.io/'>
        <img src='https://readthedocs.org/projects/eir/badge/?version=stable' alt='Documentation Status' /></a>
  
       
</p>

---

Supervised modelling, sequence generation, image generation, array output and survival analysis on genotype, tabular, sequence, image, array, and binary input data.

**WARNING:** This project is in alpha phase. Expect backwards incompatible changes and API changes between minor versions.

## What's New

- **April 2025**: More scaling related tutorials
  - [Scaling Diffusion: Image Generation from Text](https://eir.readthedocs.io/en/stable/tutorials/i_scaling/01_streaming.html) - Exploring artistic styles from historical collections
  - [Supervised Fine Tuning of GPT Style Model](https://eir.readthedocs.io/en/stable/tutorials/i_scaling/02_compute_gpt.html#e-supervised-fine-tuning-from-a-pretrained-model) - Fine tuning a pre-trained model for instruction following
- **March 2025**: Scaling tutorials added
  - [Streaming Data: Training with FineWeb](https://eir.readthedocs.io/en/stable/tutorials/i_scaling/01_streaming.html) - Train models via continuous data streaming
  - [Scaling Compute: Training a BabyGPT](https://eir.readthedocs.io/en/stable/tutorials/i_scaling/02_compute_gpt.html) - Examples of training GPT-style models

# Table of Contents
1. [Install](#install)
2. [Usage](#usage)
3. [Use Cases](#use-cases)
4. [Features](#features)
5. [Supported Inputs and Outputs](#supported-inputs-and-outputs)
6. [Related Projects](#related-projects)
7. [Citation](#citation)
8. [Acknowledgements](#acknowledgements)


## Install

### Installing EIR via `pip`

`pip install eir-dl`

**Important:** The latest version of EIR requires [Python 3.13](https://www.python.org/downloads/). Using an older version of Python will install an outdated version of EIR, which will likely be incompatible with the current documentation and might contain bugs. Please ensure you are using Python 3.13.

### Installing EIR via Container Engine

Here's an example with Docker:

```
docker build -t eir:latest https://raw.githubusercontent.com/arnor-sigurdsson/EIR/master/Dockerfile
docker run -d --name eir_container eir:latest
docker exec -it eir_container bash
```

## Usage

Please refer to the [Documentation](https://eir.readthedocs.io/en/stable/index.html) for examples and information.

## Use Cases

EIR allows for training and evaluating various deep-learning models directly from the command line. This can be useful for:

- Quick prototyping and iteration when modelling on new datasets.
- Establishing baselines to compare against other methods.
- Fitting on data sources such as large-scale genomics, where DL implementations are not commonly available.

If you are an ML/DL researcher developing new models, etc., it might not fit your use case. However, it might provide a quick baseline for comparison to the cool stuff you are developing, and there is some degree of [customization](https://eir.readthedocs.io/en/stable/tutorials/tutorial_index.html#customizing-eir) possible.

## Features

- **General**
  - Train models directly from the command line through `.yaml` configuration files.
  - Training on [genotype](https://eir.readthedocs.io/en/stable/tutorials/a_using_eir/01_basic_tutorial.html), [tabular](https://eir.readthedocs.io/en/stable/tutorials/a_using_eir/02_tabular_tutorial.html), [sequence](https://eir.readthedocs.io/en/stable/tutorials/a_using_eir/03_sequence_tutorial.html), [image](https://eir.readthedocs.io/en/stable/tutorials/a_using_eir/05_image_tutorial.html), [array](https://eir.readthedocs.io/en/stable/tutorials/a_using_eir/08_array_tutorial.html) and [binary](https://eir.readthedocs.io/en/stable/tutorials/a_using_eir/06_raw_bytes_tutorial.html) input data, with various modality-specific settings available.
  - Seamless multi-modal (e.g., combining [text + image + tabular data](https://eir.readthedocs.io/en/stable/tutorials/a_using_eir/07_multimodal_tutorial.html), or any combination of the modalities above) training.
  - Train multiple features extractors on the same data source, e.g., [combining vanilla transformer, Longformer and a pre-trained BERT variant](https://eir.readthedocs.io/en/stable/tutorials/a_using_eir/04_pretrained_sequence_tutorial.html) for text classification.
  - Support for [checkpointing and continued training](https://eir.readthedocs.io/en/stable/tutorials/e_pretraining/01_checkpointing.html), as well as [pretraining and transferring parts of trained models to new tasks](https://eir.readthedocs.io/en/stable/tutorials/e_pretraining/02_mini_foundation.html).
- **Supervised Learning**
  - Supports continuous (i.e., regression) and categorical (i.e., classification) targets.
  - [Multi-task / multi-label](https://eir.readthedocs.io/en/stable/tutorials/a_using_eir/07_multimodal_tutorial.html#appendix-b-multi-modal-multi-task-learning) prediction supported out-of-the-box.
  - Model explainability for genotype, tabular, sequence, image and array data built in.
  - Computes and graphs various evaluation metrics (e.g., RMSE, PCC and R2 for regression tasks, accuracy, ROC-AUC, etc. for classification tasks) during training.
- **Sequence Generation**
  - Supports various sequence generation tasks, including basic sequence generation, sequence to sequence transformations, and image to sequence transformations. For more information, refer to the respective tutorials: [sequence generation](https://eir.readthedocs.io/en/stable/tutorials/c_sequence_output/01_sequence_generation.html), [sequence to sequence](https://eir.readthedocs.io/en/stable/tutorials/c_sequence_output/02_sequence_to_sequence.html), [image to sequence](https://eir.readthedocs.io/en/stable/tutorials/c_sequence_output/03_image_to_sequence.html) and [tabular to sequence](https://eir.readthedocs.io/en/stable/tutorials/c_sequence_output/04_tabular_to_sequence.html).
- **Image Generation**
  - Image generation is supported. For more information, refer to the respective tutorials: [Building a Simple Image Autoencoder](https://eir.readthedocs.io/en/stable/tutorials/f_image_output/01_foundational_autoencoder.html), [Image Colorization and Super-Resolution](https://eir.readthedocs.io/en/stable/tutorials/f_image_output/02_image_coloring_and_upscaling.html) and [Guided Diffusion for Image Generation](https://eir.readthedocs.io/en/stable/tutorials/f_image_output/03_mnist_diffusion.html).
- **Array Output**
  - Supports array output tasks, such as building simple autoencoders for tasks like [MNIST Digit Generation](https://eir.readthedocs.io/en/stable/tutorials/d_array_output/01_autoencoder.html).
- **Time Series**
  - Time series inputs and outputs is possible, such as [Transformer-based Power Consumption Prediction](https://eir.readthedocs.io/en/stable/tutorials/g_time_series/01_time_series_power.html) and [Stock Price Prediction Using Transformers, One-shot and Diffusion Models](https://eir.readthedocs.io/en/stable/tutorials/g_time_series/02_time_series_stocks.html).
- **Survival Analysis**
  - Time-to-event prediction is supported as an output type, demonstrated through [Patient Survival Prediction using Free Light Chain Data](https://eir.readthedocs.io/en/stable/tutorials/h_survival_analysis/01_survival_flchain.html) and [Survival Analysis Using Cox Proportional Hazards Model](https://eir.readthedocs.io/en/stable/tutorials/h_survival_analysis/02_survival_flchain_cox.html).
- **Scaling**
  - Support for [streaming data training](https://eir.readthedocs.io/en/stable/tutorials/i_scaling/01_streaming.html) enabling models to train on datasets too large to fit in memory or with real-time data.
  - Capabilities for [scaling compute resources](https://eir.readthedocs.io/en/stable/tutorials/i_scaling/02_compute_gpt.html) to train larger language models like GPT variants.


- [Many more settings](https://eir.readthedocs.io/en/stable/api_reference.html) and configurations (e.g., augmentation, regularization, optimizers) available.

## Supported Inputs and Outputs

| Modality   | Input | Output |
|------------|:-----:|:------:|
| Genotype   | ✓     | †      |
| Tabular    | ✓     | ✓      |
| Sequence   | ✓     | ✓      |
| Image      | ✓     | ✓      |
| Array      | ✓     | ✓      |
| Binary     | ✓     |        |
| Survival   | n/a   | ✓      |

† While not directly supported, genotypes can be treated as arrays. For example see the [MNIST Digit Generation](https://eir.readthedocs.io/en/stable/tutorials/d_array_output/01_autoencoder.html) tutorial.

## Related Projects

- [EIR-auto-GP](https://github.com/arnor-sigurdsson/EIR-auto-GP): Automated genomic prediction (GP) using deep learning models with EIR.

## Citation

If you use `EIR` in a scientific publication, we would appreciate if you could use one of the following citations:

- [Deep integrative models for large-scale human genomics](https://academic.oup.com/nar/article/51/12/e67/7177885)
- [Non-linear genetic regulation of the blood plasma proteome](https://www.medrxiv.org/content/10.1101/2024.07.04.24309942v1)
- [Improved prediction of blood biomarkers using deep learning](https://www.medrxiv.org/content/10.1101/2022.10.27.22281549v1)

```
@article{10.1093/nar/gkad373,
    author    = {Sigurdsson, Arn{\'o}r I and Louloudis, Ioannis and Banasik, Karina and Westergaard, David and Winther, Ole and Lund, Ole and Ostrowski, Sisse Rye and Erikstrup, Christian and Pedersen, Ole Birger Vesterager and Nyegaard, Mette and DBDS Genomic Consortium and Brunak, S{\o}ren and Vilhj{\'a}lmsson, Bjarni J and Rasmussen, Simon},
    title     = {{Deep integrative models for large-scale human genomics}},
    journal   = {Nucleic Acids Research},
    month     = {05},
    year      = {2023}
}

@article{sigurdsson2024non,
  title={Non-linear genetic regulation of the blood plasma proteome},
  author={Sigurdsson, Arnor I and Gr{\"a}f, Justus F and Yang, Zhiyu and Ravn, Kirstine and Meisner, Jonas and Thielemann, Roman and Webel, Henry and Smit, Roelof AJ and Niu, Lili and Mann, Matthias and others},
  journal={medRxiv},
  pages={2024--07},
  year={2024},
  publisher={Cold Spring Harbor Laboratory Press}
}

@article{sigurdsson2022improved,
    author    = {Sigurdsson, Arnor Ingi and Ravn, Kirstine and Winther, Ole and Lund, Ole and Brunak, S{\o}ren and Vilhjalmsson, Bjarni J and Rasmussen, Simon},
    title     = {Improved prediction of blood biomarkers using deep learning},
    journal   = {medRxiv},
    pages     = {2022--10},
    year      = {2022},
    publisher = {Cold Spring Harbor Laboratory Press}
}
```

## Acknowledgements

Massive thanks to everyone publishing and developing the [packages](https://eir.readthedocs.io/en/stable/acknowledgements.html) this project directly and indirectly depends on.
