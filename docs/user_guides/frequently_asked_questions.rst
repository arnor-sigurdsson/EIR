Frequently Asked Questions
==========================

This guide addresses common questions and issues users encounter when
working with EIR, based on real user experiences.

Table of Contents
-----------------

1. `Attribution Analysis <#attribution-analysis>`__
2. `Model Overfitting and
   Performance <#model-overfitting-and-performance>`__
3. `Multi-modal Data Integration <#multi-modal-data-integration>`__
4. `Model Architecture <#model-architecture>`__
5. `Prediction and Configuration <#prediction-and-configuration>`__
6. `Data Handling <#data-handling>`__
7. `Validation and Testing <#validation-and-testing>`__
8. `Technical Issues and
   Performance <#technical-issues-and-performance>`__

--------------

Attribution Analysis
--------------------

Q: How do I enable attribution analysis during training?
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**A:** Add the following to your global configuration file:

.. code:: yaml

   attribution_analysis:
     compute_attributions: true
     max_attributions_per_class: 100  # Samples per class to analyze
     attributions_every_sample_factor: 4  # Compute every 4th evaluation

**Note:** Attribution calculations are computationally expensive,
especially with many output targets. Consider:

- Using higher ``attributions_every_sample_factor`` values (e.g., 4 or 8) to reduce computation
- Running attributions only on your best model after training
- Allocating more computational resources when using attributions

Q: What do the attribution values mean?
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**A:** Attribution values represent the average influence of each
feature on the model’s raw output.

-  Values are **not normalized** to sum to 1 by default
-  They show the feature importance using `Integrated Gradients <https://arxiv.org/abs/1703.01365>`_ method
-  Higher absolute values indicate stronger influence
-  Can be positive (increases output) or negative (decreases output)

To convert to percentage contributions, you could for example:

1. Check the ``feature_importance.csv`` file in the attributions folder
2. Calculate the mean attribution for each feature
3. Normalize to sum to 1 for relative importance

--------------

Model Performance
-----------------

Q: My model starts overfitting very quickly. What can I do?
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**A:** Try these strategies, organized by where they are configured:

1. **Reduce batch size** (in global configuration):

   .. code:: yaml

      # In your global configuration file
      basic_experiment:
        batch_size: 32  # Reduce from default 64

2. **Add regularization via mixing** (in global configuration):

   .. code:: yaml

      # In your global configuration file
      training_control:
        mixing_alpha: 0.2  # Mixup augmentation (0.0-1.0)

3. **Adjust learning rate and weight decay** (in global configuration):

   .. code:: yaml

      # In your global configuration file
      optimization:
        lr: 0.0001  # Reduce from default 0.0003
        wd: 0.001   # Increase weight decay from default 0.0001

4. **Enable early stopping** (in global configuration):

   .. code:: yaml

      # In your global configuration file
      training_control:
        early_stopping_patience: 10
        early_stopping_buffer: 2000  # Optional: wait before checking

5. **Increase dropout in fusion module** (in fusion configuration):

   .. code:: yaml

      # In your fusion configuration file
      model_type: mlp-residual
      model_config:
        rb_do: 0.25          # Residual block dropout (default 0.1)
        fc_do: 0.25          # Final layer dropout (default 0.1)
        stochastic_depth_p: 0.2  # Stochastic depth (default 0.1)

6. **Add dropout to output models** (in output configuration):

   .. code:: yaml

      # In your output configuration file (for tabular outputs)
      output_type_info:
        # ... other settings ...
      model_config:
        model_init_config:
          rb_do: 0.25
          fc_do: 0.25
          stochastic_depth_p: 0.2

7. **For classification tasks, use label smoothing** (in output configuration):

   .. code:: yaml

      # In your output configuration file
      output_type_info:
        target_cat_columns:
          - target_column
        cat_label_smoothing: 0.1  # Smooths one-hot labels

8. **For tabular inputs, add L1 regularization** (in input configuration):

   .. code:: yaml

      # In your tabular input configuration file
      model_config:
        model_type: tabular
        model_init_config:
          l1: 0.0001  # L1 penalty on embeddings

9. **For image inputs, use data augmentation** (in input configuration):

   .. code:: yaml

      # In your image input configuration file
      input_type_info:
        mixing_subtype: "cutmix"  # or "mixup"
        # Also uses standard augmentations by default

10. **Reduce model complexity**:

Reduce layers or hidden dimensions, fusion example shown below:

    .. code:: yaml

       # In fusion configuration
       model_config:
         layers: [1]  # Reduce from default [2]
         fc_task_dim: 128  # Reduce from default 256

Q: When should I stop training? How do I know my model is “good enough”?
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**A:** Consider these indicators:

1. **Check validation curves**: Look for plateau or degradation in
   validation performance
2. **Performance gap**: Large gap between train/validation indicates
   overfitting
3. **Task requirements**: Compare performance to your domain-specific
   needs
4. **Convergence**: If performance is still improving at max epochs,
   increase ``n_epochs``
5. **Multiple metrics**: Check ROC-AUC, MCC, and confusion matrices, not
   just loss

Look at files like ``training_curve_ROC-AUC-MACRO.pdf`` and
``training_curve_PERF-AVERAGE.pdf`` in your results folder.

Q: What is this "average performance" metric?
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**A:** EIR uses a single "average performance" metric to track overall model performance across potentially multiple tasks and output types. This metric is crucial as it controls:

- **Early stopping**: Training stops when this metric doesn't improve
- **Model checkpointing**: Models are saved based on this metric (visible in filenames like ``model_1000_perf-average=0.8547.pt``)
- **Learning rate scheduling**: LR reduction on plateau uses this metric

**How it's calculated:**

1. **For categorical outputs** (classification):

   - Default: Average of MCC, ROC-AUC-MACRO, and AP-MACRO
   - Each metric contributes equally to the average
   - Higher is better (range 0-1)

2. **For continuous outputs** (regression):

   - Default: Average of (1.0 - LOSS), PCC, and R²
   - Note: Loss is inverted so higher is better
   - Each metric contributes equally to the average

3. **For other outputs** (sequence/image generation):

   - Uses 1.0 - LOSS by default
   - Higher is better

4. **For multi-task learning**:

   - Averages across all tasks
   - Each task contributes equally regardless of output type

**Customizing the metric:**

You can choose which metrics to include in the average:

.. code:: yaml

   # In your global configuration file
   metrics:
     cat_averaging_metrics:
       - mcc
       - roc-auc-macro
       # Omit 'ap-macro' to exclude it
     con_averaging_metrics:
       - r2
       - pcc
       # Omit 'loss' to exclude it

**Example interpretation:**

- ``model_950_perf-average=0.3114.pt``: Model at iteration 950 with average performance of 0.3114
- ``model_2000_perf-average=0.8547.pt``: Model at iteration 2000 with average performance of 0.8547 (better)

**Important notes:**

- This metric is computed on the **validation set**
- A higher value always indicates better performance
- For imbalanced datasets, this averaging might mask poor performance on rare classes

--------------

Missing Data Handling
---------------------

Q: How does EIR handle missing data?
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**A:** EIR has handling for different types of missing data, both in inputs and outputs:

**Input Data - Missing Values Within a Tabular Modality:**

For partially missing data within a modality (e.g., some NaN values in tabular columns):

- **Continuous columns**: Imputed with the mean from the *training set* (e.g. will be 0 if data is already mean-normalized before being passed to EIR)
- **Categorical columns**: Encoded as a special ``__NULL__`` category
- **No manual imputation needed** - EIR handles this automatically, but you can preprocess if desired

**Input Data - Completely Missing Modalities:**

When an entire modality is missing for a sample (e.g., no image for a specific ID):

- **Tabular**: Uses the within-modality strategy above
- **Images**: Filled with random noise (Gaussian distribution)
- **Sequences/Text**: Filled with padding tokens
- **Omics**: Filled with zeros (``0`` values for the one-hot encoding)
- **Arrays**: Filled with random noise (Gaussian distribution)

**Output Data - Missing Target Values:**

- **EIR excludes NaN values from loss computation** - they don't contribute to backpropagation
- **Supports partial outputs**: Can have some target columns missing for specific samples
- **No imputation needed**: The model learns only from available labels

**Best Practices:**

1. **Preprocessing**: You may still want to filter features/samples with excessive missing values

**Example: Multi-modal with Missing Data**

.. csv-table:: Tabular Data
   :header: "ID", "Feature1", "Feature2", "Feature3"
   :widths: 15, 15, 15, 15

   "sample1", 1.5, NaN, "A"
   "sample2", 2.3, 0.8, "B"
   "sample3", NaN, 1.2, NaN

.. code-block:: text

   # images folder
   sample1.jpg
   sample2.jpg
   sample3.jpg

EIR will automatically handle the NaN values in Feature2/Feature3 and the missing image for sample2.


Model Architecture
------------------

Q: What’s the difference between mlp-residual and regular MLP?
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**A:** The ``mlp-residual`` model uses residual blocks with:

- Skip connections
- Layer normalization via RMSNorm
- GELU activation
- Stochastic depth option
- LayerScale for better training stability

Q: How do I interpret the model architecture?
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**A:** Check ``model_info.txt`` in your experiment folder.

--------------

Prediction and Configuration
----------------------------

Q: Why does ``eirpredict`` require the global_configs file?
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**A:** The global configuration contains settings needed for prediction:

- Batch size (might want to increase for faster inference)
- Attribution settings (if computing on test set)
- Dataloader workers
- Other runtime parameters

These aren’t just training parameters - they affect how predictions are
computed.

Q: How do I predict on data without labels?
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**A:** Set ``output_source: null`` in your output configuration:

.. code:: yaml

   output_info:
     output_name: my_output
     output_source: null  # Instead of path to labels
     output_type: tabular
   output_type_info:
     target_cat_columns:
       - target_column

Q: Which model checkpoint should I use for predictions?
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**A:** Generally use the model with best validation performance:

- Check the filename: ``model_950_perf-average=0.3114.pt``
- The number (950) is the iteration
- ``perf-average`` shows the validation performance
- Higher is better for most metrics

--------------

Data Handling
-------------

Q: How should I format time series data?
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**A:** Time series data can for example be formatted as sequences:

.. csv-table:: Sequence Data
   :header: "ID", "Sequence"
   :widths: 15, 40

   "sample1", "val1 val2 val3 val4 val5"
   "sample2", "val1 val2 val3 val4 val5"

Configuration example:

.. code:: yaml

   input_type_info:
     max_length: 48
     split_on: " "
     sampling_strategy_if_longer: "from_start"

Note you can also have them as separate ``.txt`` files, filename being the
sample ID and content being the sequence values.

--------------

Validation and Testing
----------------------

Q: How do I ensemble multiple model runs?
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**A:** For better stability, train multiple models with different seeds:

.. code:: bash

   EIR_SEED=0 eirtrain ...
   EIR_SEED=1 eirtrain ...
   EIR_SEED=2 eirtrain ...

Then average predictions across models.

--------------

Technical Issues and Performance
--------------------------------

Q: Attribution analysis makes training very slow. What can I do?
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**A:** Several strategies:

1. **Increase sampling interval**:

   .. code:: yaml

      attributions_every_sample_factor: 8  # or higher

2. **Reduce samples analyzed**:

   .. code:: yaml

      max_attributions_per_class: 50  # instead of 100+

3. **Run post-training**: Train without attributions, then run
   ``eirpredict`` with attributions enabled

4. **Allocate more resources**: Increase CPU/RAM allocation on your
   cluster

Q: How do I reduce training time?
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**A:** Try these optimizations:

1. **Enable model compilation on GPU/CUDA devices**:

   .. code:: yaml

      # In your global configuration file
      model:
        compile_model: true

2. **Use mixed precision training** (especially on modern GPUs):

   .. code:: yaml

      # In your global configuration file
      accelerator:
        precision: "16-mixed"  # or "bf16-mixed" for newer GPUs

3. **Load data into memory** (if you have enough RAM):

   .. code:: yaml

      # In your global configuration file
      basic_experiment:
        memory_dataset: true

4. **Increase dataloader workers** (for CPU-bound data loading):

   .. code:: yaml

      # In your global configuration file
      basic_experiment:
        dataloader_workers: 8  # Adjust based on CPU cores

5. **Use gradient accumulation** (simulate larger batches without more memory):

   .. code:: yaml

      # In your global configuration file
      optimization:
        gradient_accumulation_steps: 4  # Effective batch = batch_size * 4

6. **Reduce evaluation frequency**:

   .. code:: yaml

      # In your global configuration file
      evaluation_checkpoint:
        sample_interval: 500  # instead of 200
        checkpoint_interval: 500

7. **Feature selection**: Use fewer input features based on prior knowledge or attributions

8. **Smaller models**: Reduce layers or hidden dimensions in fusion/output configs

9. **Early stopping**: Stop when validation performance plateaus

**Quick wins for GPU training:**

- Set ``compile_model: true`` and ``precision: "16-mixed"``
- Use ``memory_dataset: true`` if your dataset fits in RAM
- Increase ``dataloader_workers`` to 2-4

**Note:** Model compilation may not work with all architectures.
Mixed precision can slightly affect model accuracy but usually
provides significant speedup with minimal impact.

--------------

Need More Help?
---------------

-  Check the `official documentation <https://eir.readthedocs.io/>`__
-  Review tutorials for specific use cases
-  For genomics-specific tasks, consider
   `EIR-auto-GP <https://github.com/arnor-sigurdsson/EIR-auto-GP>`__
-  Examine the generated ``model_info.txt`` for architecture details
