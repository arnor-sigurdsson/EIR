Supervised Learning
===================

Learn the fundamentals of EIR with classification and regression tasks.


Single Modality Tutorials
--------------------------

.. toctree::
    :maxdepth: 1

    a_using_eir/01_basic_tutorial
    a_using_eir/02_tabular_tutorial
    a_using_eir/03_sequence_tutorial
    a_using_eir/05_image_tutorial
    a_using_eir/06_raw_bytes_tutorial
    a_using_eir/08_array_tutorial

Advanced Techniques
-------------------

.. toctree::
    :maxdepth: 1

    a_using_eir/04_pretrained_sequence_tutorial
    a_using_eir/07_multimodal_tutorial

Generative Modeling
===================

Learn to generate new data with EIR's generative capabilities.

Sequence Generation
-------------------

Generate text, DNA sequences, and other sequential data.

.. toctree::
    :maxdepth: 1

    c_sequence_output/01_sequence_generation
    c_sequence_output/02_sequence_to_sequence
    c_sequence_output/03_image_to_sequence
    c_sequence_output/04_tabular_to_sequence

Image Generation
----------------

Create and manipulate images with autoencoders and diffusion models.

.. toctree::
    :maxdepth: 1

    f_image_output/01_foundational_autoencoder
    f_image_output/02_image_coloring_and_upscaling
    f_image_output/03_mnist_diffusion

Array Generation
----------------

Generate multi-dimensional data and build autoencoders.

.. toctree::
    :maxdepth: 1

    d_array_output/01_autoencoder

Specialized Applications
========================

Domain-specific tutorials for specialized use cases.

Time Series Forecasting
-----------------------

Predict future values in temporal data.

.. toctree::
    :maxdepth: 1

    g_time_series/01_time_series_power
    g_time_series/02_time_series_stocks

Survival Analysis
-----------------

Model time-to-event data and survival probabilities.

.. toctree::
    :maxdepth: 1

    h_survival_analysis/01_survival_flchain
    h_survival_analysis/02_survival_flchain_cox

Advanced Topics
===============

Streaming, scaling, and customization.

Model Management
----------------

.. toctree::
    :maxdepth: 1

    e_pretraining/01_checkpointing
    e_pretraining/02_mini_foundation

Scaling
-------

These tutorials demonstrate how to use EIR for larger-scale experiments,
while generally tutorials can be run on a single laptop, these tutorials
generally require a GPU.

.. toctree::
    :maxdepth: 1

    i_scaling/01_streaming
    i_scaling/02_compute_gpt
    i_scaling/03_diffusion

Customization
-------------

.. toctree::
    :maxdepth: 1

    b_customizing_eir/01_customizing_fusion

**Next Steps:**

* Check the :doc:`../api/api_reference` for detailed configuration options