.. _sequence-configurations:

Sequence Data Configuration
===========================

Complete configuration guide for sequential data including text, DNA sequences, and time series.

.. contents::
   :local:
   :depth: 2

Overview
--------

Sequence data in EIR handles data such as:

* **Text data** - Natural language, documents, reviews
* **Biological sequences** - DNA, RNA, protein sequences
* **Time series** - Sequential measurements over time

Quick Example
-------------

.. code-block:: yaml

   input_info:
     input_source: "my/sequence/data/"
     input_name: "dna_sequence"
     input_type: "sequence"
   input_type_info:
     vocab_file: "dna_vocab.json"
     max_length: 1024
   model_config:
     model_type: "sequence-default"
     model_init_config:
       embedding_dim: 128
       num_heads: 8
       num_layers: 6

Input Data Configuration
------------------------

Base Configuration
^^^^^^^^^^^^^^^^^^

.. autoclass:: eir.setup.schemas.SequenceInputDataConfig

Model Selection
^^^^^^^^^^^^^^^

.. autoclass:: eir.models.input.sequence.transformer_models.SequenceModelConfig

Available Feature Extractors
-----------------------------

Built-in Sequence Models
^^^^^^^^^^^^^^^^^^^^^^^^

.. autoclass:: eir.models.input.sequence.transformer_models.BasicTransformerFeatureExtractorModelConfig

External Sequence Models
^^^^^^^^^^^^^^^^^^^^^^^^

For pre-trained language models (BERT, GPT, etc.), please refer to :ref:`external-sequence-models` for detailed configuration options.

Interpretation Support
----------------------

.. autoclass:: eir.setup.schemas.BasicInterpretationConfig