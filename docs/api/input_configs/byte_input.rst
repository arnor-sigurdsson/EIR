.. _byte-configurations:

Byte Data Configuration
=======================

Complete configuration guide for binary and byte-level data processing.

.. contents::
   :local:
   :depth: 2

Overview
--------

Byte data in EIR handles raw binary data and byte sequences:

* **File analysis** - Binary file content analysis
* **Network data** - Packet analysis, protocol detection
* **Raw text** - Byte-level text processing
* **Binary sequences** - Any sequence of bytes

Quick Example
-------------

.. code-block:: yaml

   input_info:
     input_source: "my/binary/data/folder/"
     input_name: "binary_file"
     input_type: "bytes"
   input_type_info:
     vocab_file: "byte_vocab.json"
     max_length: 2048
   model_config:
     model_type: "sequence-default"
     model_init_config:
       embedding_dim: 64
       num_heads: 4

Input Data Configuration
------------------------

Base Configuration
^^^^^^^^^^^^^^^^^^

.. autoclass:: eir.setup.schemas.ByteInputDataConfig

Available Feature Extractors
-----------------------------

Byte data typically uses sequence-based feature extractors. See :doc:`sequence_input` for detailed configuration options of sequence models that can process byte data.