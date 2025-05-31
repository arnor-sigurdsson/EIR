.. _sequence-output-configurations:

Sequence Output Configuration
=============================

Complete configuration guide for sequence generation and prediction tasks.

.. contents::
   :local:
   :depth: 2

Overview
--------

Sequence outputs handle sequential data generation including:

* **Text generation** - Natural language generation, summarization
* **Sequence-to-sequence** - Translation, transformation tasks
* **DNA generation** - Biological sequence synthesis
* **Time series forecasting** - Future value prediction

Quick Example
-------------

.. code-block:: yaml

   output_info:
     output_source: "my_sequence_output_folder/"
     output_name: "generated_text"
     output_type: "sequence"
   output_type_info:
     vocab_file: "output_vocab.json"
     max_length: 512
   model_config:
     model_type: "sequence"
     model_init_config:
       embedding_dim: 256
       num_heads: 8

Output Type Configuration
-------------------------

.. autoclass:: eir.setup.schema_modules.output_schemas_sequence.SequenceOutputTypeConfig

Output Module Configuration
---------------------------

.. autoclass:: eir.models.output.sequence.sequence_output_modules.SequenceOutputModuleConfig

Output Sampling Configuration
-----------------------------

.. autoclass:: eir.setup.schema_modules.output_schemas_sequence.SequenceOutputSamplingConfig