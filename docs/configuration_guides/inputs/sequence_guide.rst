Sequence Data Guide
===================

Ready-to-use configurations for sequence data analysis
using Transformer-based models in EIR.

- **Supported data types:** Text (NLP), protein/peptide sequences, DNA/RNA,
  time series, and other discrete token sequences
- **Data format:** A folder with ``.txt`` files (filename is the ID) or
  a ``.csv`` file with columns ``"ID"`` and ``"Sequence"``
- **Models:** Built-in transformer (``sequence-default``),
  external pretrained models (BERT, RoBERTa, etc., see :ref:`external-sequence-models`).

.. note::
   **First step:** Copy the :doc:`../guides_index` global configuration as your ``globals.yaml``

.. contents::
   :local:
   :depth: 2

Quick Start
-----------

- **Use cases:** Sequence classification (sentiment, protein function), regression (binding affinity), or generation
- **Data requirements:** Sequence data in text files or CSV format, labels for supervised tasks

**Files needed:**

.. code-block:: yaml
   :caption: inputs.yaml

   input_info:
     input_source: data/protein_sequences/      # Path to folder with .txt files or .csv file
     input_name: sequence
     input_type: sequence

   input_type_info:
     max_length: 512                           # Sequence length (int, 'max', or 'average')

     # Split on characters for proteins/DNA
     # ("" for char-level, " " for words,
     # null for no splitting e.g. when using BPE tokenizer)
     split_on: ""

     tokenizer: null                           # No tokenizer (see advanced options below)
     min_freq: 2                               # Minimum token frequency for vocabulary

   model_config:
     model_type: sequence-default              # Built-in transformer for sequences
     model_init_config:
       embedding_dim: 128                      # Token embedding dimension
       num_layers: 4                           # Number of transformer layers
       num_heads: 8                            # Number of attention heads per layer
       dropout: 0.10                           # Dropout rate

.. note::
    The ``input_source`` can be:

    - A directory of ``.txt`` files where the filename (without extension) is the sample ID
    - A ``.csv`` file with columns ``"ID"`` and ``"Sequence"``

    For protein/DNA sequences, use ``split_on: ""`` for character-level tokenization.
    For natural language, use ``split_on: " "`` for word-level tokenization.

    Alternatively, set ``split_on: null`` for no splitting, and use the
    `BPE tokenizer <https://en.wikipedia.org/wiki/Byte_pair_encoding>`_
    (``tokenizer: "bpe"``) for an adaptive vocabulary.


.. code-block:: yaml
   :caption: outputs.yaml

   output_info:
     output_name: sequence_label
     output_source: data/labels.csv           # Must contain "ID" column + targets
     output_type: tabular

   output_type_info:
     target_cat_columns:
       - Function_Class                        # Categorical target (e.g., protein function)
     target_con_columns:
       - Binding_Affinity                      # Continuous target (optional)

**Run command:**

.. code-block:: bash

   eirtrain --global_configs globals.yaml \
            --input_configs inputs.yaml \
            --output_configs outputs.yaml

About Sequence Models
---------------------

**Full model configuration with all available parameters:**

.. code-block:: yaml
   :caption: Advanced sequence configuration

   model_config:
     model_type: sequence-default
     model_init_config:
       # Architecture parameters
       embedding_dim: 128                      # Dimension of token embeddings
       num_layers: 6                           # Number of transformer layers
       num_heads: 8                            # Number of attention heads
       dropout: 0.10                           # Dropout rate in transformer layers

       # Advanced architecture options
       dim_feedforward: 512                    # Feedforward network dimension

       # Attention mechanisms
       window_size: null                       # Local attention window (null = full attention)

As always, please refer to the
API documentation :ref:`sequence-configurations` for
the full list of available parameters and more in-depth explanations.

Common Use Cases
----------------

Natural Language Processing
^^^^^^^^^^^^^^^^^^^^^^^^^^^

For text classification, sentiment analysis, or document classification:

.. code-block:: yaml
   :caption: Text classification setup

   input_type_info:
     max_length: 512
     split_on: " "                             # Split on whitespace for words
     tokenizer: "basic_english"                # English text normalization
     min_freq: 5                               # Filter rare words

Biological Sequences
^^^^^^^^^^^^^^^^^^^^

For protein, peptide, or DNA sequence analysis:

.. code-block:: yaml
   :caption: Protein sequence setup

   input_type_info:
     max_length: 1024                          # Typical protein length
     split_on: ""                              # Character-level tokenization
     tokenizer: null                           # No additional tokenization
     min_freq: 1                               # Keep all amino acids/nucleotides

Time Series Data
^^^^^^^^^^^^^^^^

For sequential numeric data represented as text
(assumes they have e.g. been binned/discretized beforehand):

.. code-block:: yaml
   :caption: Time series setup

   input_type_info:
     max_length: "average"                     # Use average sequence length
     split_on: ","                             # Split on delimiter
     tokenizer: null                           # No tokenization
     sampling_strategy_if_longer: "uniform"    # Random sampling for long sequences

Advanced Tokenization
---------------------

**BPE (Byte Pair Encoding) Tokenization:**

For subword tokenization, particularly useful for handling out-of-vocabulary words:

.. code-block:: yaml
   :caption: BPE tokenizer configuration

   input_type_info:
     tokenizer: "bpe"
     adaptive_tokenizer_max_vocab_size: 10000  # Maximum vocabulary size
     vocab_file: null                          # Will be trained on your data
     split_on: null                            # BPE handles splitting internally


**Custom Vocabulary:**

Using a pre-defined vocabulary file:

.. code-block:: yaml
   :caption: Custom vocabulary setup

   input_type_info:
     vocab_file: "data/custom_vocab.json"     # JSON file with token->id mapping

.. note::
      The vocab file is a optional text file containing pre-defined vocabulary to use
      for the training. If this is not passed in, the framework will automatically
      build the vocabulary from the training data. Passing in a vocabulary file is
      therefore useful if (a) you want to manually specify / limit the vocabulary used
      and/or (b) you want to save time by pre-computing the vocabulary.

      Here, there are two formats supported:

      - A ``.json`` file containing a dictionary with the vocabulary as keys and
        the corresponding token IDs as values. For example:
        ``{"the": 0, "cat": 1, "sat": 2, "on": 3, "the": 4, "mat": 5}``

      - A ``.json`` file with the results of training and saving the vocabulary of
        a Huggingface BPE tokenizer. This is the file create by calling
        ``hf_tokenizer.save()``. This is only valid when using the ``bpe`` tokenizer.



Sequence Length Strategies
--------------------------

**Dynamic Length Calculation:**

.. code-block:: yaml
   :caption: Dynamic length options

   input_type_info:
     max_length: "max"                         # Use longest sequence in dataset
     # OR
     max_length: "average"                     # Use average length
     # OR
     max_length: 512                           # Fixed length

**Handling Long Sequences:**

.. code-block:: yaml
   :caption: Long sequence handling

   input_type_info:
     sampling_strategy_if_longer: "uniform"   # Random sampling for training
     # OR
     sampling_strategy_if_longer: "from_start" # Always truncate from beginning

.. note::
   Validation and test sets always use ``"from_start"`` for consistency,
   regardless of the training strategy.

External Pretrained Models
--------------------------

For leveraging pretrained language models:

.. code-block:: yaml
   :caption: Using pretrained BERT

   model_config:
     model_type: "bert-base-uncased"           # Hugging Face model name
     pretrained_model: true                    # Use pretrained weights
     model_init_config:
       num_labels: 2                           # Number of output classes

See :ref:`external-sequence-models` for the full list of supported models.


Attribution Analysis
--------------------

Enable feature importance analysis to understand which parts of sequences
contribute most to predictions:

.. code-block:: yaml
   :caption: Attribution analysis setup (in globals.yaml)

   attribution_analysis:
     compute_attributions: true
     max_attributions_per_class: 100          # Samples per class to analyze
     attributions_every_sample_factor: 4      # Compute every 4th evaluation

This uses `Integrated Gradients <https://arxiv.org/abs/1703.01365>`_ to compute
token-level importance scores, helping you understand model decisions.

Complete Configuration Examples
-------------------------------

**Protein Function Prediction:**

.. code-block:: yaml
   :caption: Complete protein classification setup

   # inputs.yaml
   input_info:
     input_source: data/protein_sequences/
     input_name: protein_seq
     input_type: sequence
   input_type_info:
     max_length: 1024
     split_on: ""                              # Character-level for amino acids
     min_freq: 1                               # Keep all amino acids
   model_config:
     model_type: sequence-default
     model_init_config:
       embedding_dim: 128
       num_layers: 4
       num_heads: 8
       dropout: 0.10

   # outputs.yaml
   output_info:
     output_name: protein_function
     output_source: data/protein_labels.csv
     output_type: tabular
   output_type_info:
     target_cat_columns:
       - Enzyme_Class
       - Subcellular_Location

**Sentiment Analysis:**

.. code-block:: yaml
   :caption: Complete sentiment analysis setup

   # inputs.yaml
   input_info:
     input_source: data/reviews.csv           # CSV with ID and Sequence columns
     input_name: review_text
     input_type: sequence
   input_type_info:
     max_length: 512
     split_on: " "                             # Word-level tokenization
     tokenizer: "basic_english"                # Text normalization
     min_freq: 5                               # Filter rare words
   model_config:
     model_type: sequence-default
     model_init_config:
       embedding_dim: 256
       num_layers: 6
       num_heads: 8
       dropout: 0.10

   # outputs.yaml
   output_info:
     output_name: sentiment
     output_source: data/sentiment_labels.csv
     output_type: tabular
   output_type_info:
     target_cat_columns:
       - Sentiment                             # Positive/Negative