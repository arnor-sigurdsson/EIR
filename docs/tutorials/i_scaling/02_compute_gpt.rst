.. _i-scaling-compute-gpt:

.. role:: raw-html(raw)
    :format: html

02 – Scaling Compute: Training a BabyGPT
========================================

In this tutorial, we will examine how we can scale up `EIR` to train a baby version
of the GPT model, streaming data from the FineWeb dataset for model training.

.. note::
    This tutorial assumes you are familiar with the basics of `EIR`.
    While not required, it's recommended to have gone through the basic tutorials first.

.. note::
    See :ref:`i-scaling-streaming-data` and :ref:`streaming-data-guide`
    for more information on streaming data in EIR.

A - Overview
------------

This tutorial largely follows the same approach as in :ref:`i-scaling-streaming-data`.
Again, we will be using a WebSocket server to stream data, but this time we will be
focusing on scaling compute by training a larger model.

B - Setting Up
--------------

Here's the folder structure we'll be working with:

.. literalinclude:: ../tutorial_files/i_scaling/02_scaling_compute/commands/tutorial_folder.txt
    :language: console

The global config specifies basic training and below are highlighted a couple
related to scaling:

.. literalinclude:: ../tutorial_files/i_scaling/02_scaling_compute/globals.yaml
    :language: yaml
    :caption: globals.yaml
    :emphasize-lines: 8-10,17-24

Now, the optimization parameters are not directly related to scaling, but follow
common practice used to train large language models such as GPT2. However, we
do specify that we want to compile the model and train with a ``bf16-mixed`` precision,
which is becoming common practice for training large models. Additionally, many
options related to e.g. hardware are set to ``auto`` (which is the default),
which allows the framework to automatically select the best options for the
compute environment being used.
Note a lot of this functionality is
taking advantage of the excellent done by the folks at `PyTorch <https://pytorch.org/>`__
and `Fabric <https://lightning.ai/docs/fabric/stable/>`__.

You might also notice the ``streaming_setup_samples`` option there, read a
bit further down to when we are discussing the ``output.yaml`` for more
details.

For fusion, we use a simple pass-through configuration since we're only doing
sequence generation:

.. literalinclude:: ../tutorial_files/i_scaling/02_scaling_compute/fusion.yaml
    :language: yaml
    :caption: fusion.yaml

Compared to the streaming tutorial, we can see here how we are increasing the
maximum sequence length used by the model (now 512), as well as a bunch of parameters
in the model configuration.

.. literalinclude:: ../tutorial_files/i_scaling/02_scaling_compute/output.yaml
    :language: yaml
    :caption: output.yaml
    :emphasize-lines: 10,15-22

You might also notice something new here, we are using the ``tokenizer`` option to point
to the file ``fineweb_tokenizer.json``. This is a tokenizer file created
with the `tokenizers <https://huggingface.co/docs/tokenizers/index>`__ library,
specifically the BPE tokenizer, trained on 0.5m samples from the FineWeb dataset.
You can download the tokenizer file from
`this link. <https://drive.google.com/file/d/10nqy1z3wqt-KHOlYw_t-UgfcEMdFSySO>`__.
Now, we can also omit this and allow ``EIR`` to train the optimizer from scratch,
this is where the ``streaming_setup_samples`` option comes in. This option controls
how many samples are collected from the streaming server to use for setup (e.g.,
training the tokenizer, estimating means for imputation, etc.). However, training
the tokenizer can take a while, so to speed things up a bit, we are using
the pre-trained tokenizer.

C - Training
------------

Before starting training, we need to ensure our streaming server is running.
The server will serve chunks of text from the FineWeb dataset. Once it's
running, we can start training:

.. literalinclude:: ../tutorial_files/i_scaling/02_scaling_compute/commands/STREAMING_SEQUENCE_GENERATION.txt
    :language: console

Now, to train this, you almost certainly need a GPU. For this tutorial,
the model was trained for around 2 hours on two H100 GPUs.

At iteration 500:

.. literalinclude:: ../tutorial_files/i_scaling/02_scaling_compute/figures/auto_generated_iter_500.txt
    :language: console
    :caption: Auto-generated sequence at iteration 500

.. literalinclude:: ../tutorial_files/i_scaling/02_scaling_compute/figures/manual_generated_iter_500.txt
    :language: console
    :caption: Manually generated sequence at iteration 500

By iteration 52000, we can see improvement:

.. literalinclude:: ../tutorial_files/i_scaling/02_scaling_compute/figures/auto_generated_iter_52000.txt
    :language: console
    :caption: Auto-generated sequence at iteration 52000

.. literalinclude:: ../tutorial_files/i_scaling/02_scaling_compute/figures/manual_generated_iter_52000.txt
    :language: console
    :caption: Manually generated sequence at iteration 52000

Here's the training curve showing our progress:

.. image:: ../tutorial_files/i_scaling/02_scaling_compute/figures/training_curve_LOSS.png
    :width: 100%
    :align: center

D - Complete Server Implementation
----------------------------------

Here's the complete implementation of our streaming server, which you can use
as a reference for implementing your own:

.. literalinclude:: ../../doc_modules/i_scaling/text_streamer.py
    :language: python
    :caption: text_streamer.py

F - Conclusion
--------------

In this tutorial we explored how to scale up `EIR`
to train a baby version of the GPT model,
streaming data from the FineWeb dataset for model training.

Thank you for reading!