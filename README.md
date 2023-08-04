<p align="center">
  <img src="docs/source/_static/img/EIR_logo.png">
</p>

<p align="center">
  <a href="LICENSE" alt="License">
        <img src="https://img.shields.io/badge/License-APGL-5B2D5B.svg" /></a>
  
  <a href="https://github.com/arnor-sigurdsson/EIR#citation" alt="Citation">
        <img src="https://img.shields.io/badge/Papers-View%20Here-5F9EA0.svg" /></a>
  
  <a href="https://www.python.org/downloads/" alt="Python">
        <img src="https://img.shields.io/badge/python-3.11-blue.svg" /></a>
  
  <a href="https://pypi.org/project/eir-dl/" alt="Python">
        <img src="https://img.shields.io/pypi/v/eir-dl.svg" /></a>
  
  <a href="https://codecov.io/gh/arnor-sigurdsson/EIR" alt="Coverage">
        <img src="https://codecov.io/gh/arnor-sigurdsson/EIR/branch/master/graph/badge.svg" /></a>
  
  <a href='https://eir.readthedocs.io/'>
        <img src='https://readthedocs.org/projects/eir/badge/?version=latest' alt='Documentation Status' /></a>
  
       
</p>

---

Supervised modelling and sequence generation on genotype, tabular, sequence, image, array, and binary data.

**WARNING:** This project is in alpha phase. Expect backwards incompatible changes and API changes.

## Install

### Installing EIR via `pip`

`pip install eir-dl`

**Important:** The latest version of EIR supports [Python 3.11](https://www.python.org/downloads/). Using an older version of Python will install a outdated version of EIR, which likely be incompatible with the current documentation and might contain bugs. Please ensure that you are installing EIR in a Python 3.11 environment.

### Installing EIR via Container Engine

Here's an example with Docker:

```
docker build -t eir:latest https://raw.githubusercontent.com/arnor-sigurdsson/EIR/master/Dockerfile
docker run -d --name eir_container eir:latest
docker exec -it eir_container bash
```

## Usage

Please refer to the [Documentation](https://eir.readthedocs.io/en/latest/index.html) for examples and information.

## Use Cases

EIR allows for training and evaluating various deep-learning models directly from the command line. This can be useful for:

- Quick prototyping and iteration when doing supervised modelling or sequence generation on new datasets.
- Establishing baselines to compare against other methods.
- Fitting on data sources such as large-scale genomics, where DL implementations are not commonly available.

If you are an ML/DL researcher developing new models, etc., it might not fit your use case. However, it might provide a quick baseline for comparison to the cool stuff you are developing, and there is some degree of [customization](https://eir.readthedocs.io/en/latest/tutorials/tutorial_index.html#customizing-eir) possible.

## Features

- **General**
  - Train models directly from the command line through `.yaml` configuration files.
  - Training on [genotype](https://eir.readthedocs.io/en/latest/tutorials/a_using_eir/01_basic_tutorial.html), [tabular](https://eir.readthedocs.io/en/latest/tutorials/a_using_eir/02_tabular_tutorial.html), [sequence](https://eir.readthedocs.io/en/latest/tutorials/a_using_eir/03_sequence_tutorial.html), [image](https://eir.readthedocs.io/en/latest/tutorials/a_using_eir/05_image_tutorial.html), [array](https://eir.readthedocs.io/en/latest/tutorials/a_using_eir/08_array_tutorial.html) and [binary](https://eir.readthedocs.io/en/latest/tutorials/a_using_eir/06_raw_bytes_tutorial.html) input data, with various modality-specific settings available.
  - Seamless multi-modal (e.g., combining [text + image + tabular data](https://eir.readthedocs.io/en/latest/tutorials/a_using_eir/07_multimodal_tutorial.html), or any combination of the modalities above) training.
  - Train multiple features extractors on the same data source, e.g., [combining vanilla transformer, Longformer and a pre-trained BERT variant](https://eir.readthedocs.io/en/latest/tutorials/a_using_eir/04_pretrained_sequence_tutorial.html) for text classification.
- **Supervised Learning**
  - Supports continuous (i.e., regression) and categorical (i.e., classification) targets.
  - [Multi-task / multi-label](https://eir.readthedocs.io/en/latest/tutorials/a_using_eir/07_multimodal_tutorial.html#appendix-b-multi-modal-multi-task-learning) prediction supported out-of-the-box.
  - Model explainability for genotype, tabular, sequence, image and array data built in.
  - Computes and graphs various evaluation metrics (e.g., RMSE, PCC and R2 for regression tasks, accuracy, ROC-AUC, etc. for classification tasks) during training.
- **Sequence generation**
  - Supports various sequence generation tasks, including basic sequence generation, sequence to sequence transformations, and image to sequence transformations. For more information, refer to the respective tutorials: [sequence generation](https://eir.readthedocs.io/en/latest/tutorials/c_sequence_output/01_sequence_generation.html), [sequence to sequence](https://eir.readthedocs.io/en/latest/tutorials/c_sequence_output/02_sequence_to_sequence.html), and [image to sequence](https://eir.readthedocs.io/en/latest/tutorials/c_sequence_output/03_image_to_sequence.html).

- [Many more settings](https://eir.readthedocs.io/en/latest/api_reference.html) and configurations (e.g., augmentation, regularization, optimizers) available.

## Related Projects

- [EIR-auto-GP](https://github.com/arnor-sigurdsson/EIR-auto-GP): Automated genomic prediction (GP) using deep learning models with EIR.

## Citation

If you use `EIR` in a scientific publication, we would appreciate if you could use one of the following citations:

```
@article{10.1093/nar/gkad373,
    author    = {Sigurdsson, Arn{\'o}r I and Louloudis, Ioannis and Banasik, Karina and Westergaard, David and Winther, Ole and Lund, Ole and Ostrowski, Sisse Rye and Erikstrup, Christian and Pedersen, Ole Birger Vesterager and Nyegaard, Mette and DBDS Genomic Consortium and Brunak, S{\o}ren and Vilhj{\'a}lmsson, Bjarni J and Rasmussen, Simon},
    title     = {{Deep integrative models for large-scale human genomics}},
    journal   = {Nucleic Acids Research},
    month     = {05},
    year      = {2023}
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

Massive thanks to everyone publishing and developing the [packages](https://eir.readthedocs.io/en/latest/acknowledgements.html) this project directly and indirectly depends on.
