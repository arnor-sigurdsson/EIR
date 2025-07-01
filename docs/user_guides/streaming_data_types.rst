.. _streaming-data-types:

.. role:: raw-html(raw)
    :format: html

Streaming Data Types
====================

This guide is intended as a reference guide
for adding support for different data types
when streaming. For


Data Format Specifications
--------------------------

The streaming protocol requires
specific data structures for different modalities,
with distinct patterns for inputs and outputs.
But first, each sample in a batch must follow this structure:

.. code-block:: python

    {
        "inputs": {
            "modality_name": data,
            ...
        },
        "target_labels": {
            "output_name": data,
            ...
        },
        "sample_id": str  # Unique identifier for the sample
    }

Input Modalities
~~~~~~~~~~~~~~~~

Sequence Input
^^^^^^^^^^^^^^
Direct string representation:

.. code-block:: python

    "sequence_data": "raw text sequence"

Tabular Input
^^^^^^^^^^^^^
Dictionary of column names and values:

.. code-block:: python

    "tabular_data": {
        "column1": "categorical_value",
        "column2": 0.5  # numeric value
    }

Array Input
^^^^^^^^^^^
Base64 encoded numpy array:

.. code-block:: python

    "array_data": base64.b64encode(
        np.array([[0.1, 0.2], [0.3, 0.4]], dtype=np.float32).tobytes()
    ).decode("utf-8")

Image Input
^^^^^^^^^^^
Base64 encoded PNG image:

.. code-block:: python

    "image_data": base64.b64encode(
        # Convert PIL Image to bytes
        image_to_bytes(Image.fromarray(array))
    ).decode("utf-8")

Omics Input
^^^^^^^^^^^
Base64 encoded boolean array:

.. code-block:: python

    "omics_data": base64.b64encode(
        np.array([[True, False], [False, True]], dtype=np.bool_).tobytes()
    ).decode("utf-8")

Output Modalities
~~~~~~~~~~~~~~~~~

Tabular Output
^^^^^^^^^^^^^^
Nested dictionary structure with target name:

.. code-block:: python

    "test_output": {
        "target_column": value
    }

Array Output
^^^^^^^^^^^^
Nested dictionary with array name and base64 encoded data:

.. code-block:: python

    "output_array": {
        "output_array": base64.b64encode(
            np.array(...).tobytes()
        ).decode("utf-8")
    }

Image Output
^^^^^^^^^^^^
Nested dictionary with image name and base64 encoded data:

.. code-block:: python

    "output_image": {
        "output_image": base64.b64encode(
            image_to_bytes(Image.fromarray(array))
        ).decode("utf-8")
    }

Sequence Output
^^^^^^^^^^^^^^^
Nested dictionary with sequence name and string:

.. code-block:: python

    "output_sequence": {
        "output_sequence": "generated text sequence"
    }

Survival Output
^^^^^^^^^^^^^^^
Dictionary with required survival columns:

.. code-block:: python

    "output_survival": {
        "Event": "0",  # str representation of origin
        "Time": 0.5   # float value
    }

Complete Example
----------------

Here's a complete example showing how to structure a sample with multiple modalities:

.. code-block:: python

    def generate_sample() -> dict:
        # Prepare input data
        sequence = "example sequence"
        omics = np.random.rand(4, 100).astype(np.bool_)
        array_input = np.random.rand(10, 5).astype(np.float32)
        image = np.random.randint(0, 255, (16, 16, 3), dtype=np.uint8)

        # Prepare output data
        test_target = 1000
        array_output = np.random.rand(5, 3).astype(np.float32)
        image_output = np.random.randint(0, 255, (16, 16, 3), dtype=np.uint8)
        sequence_output = "generated sequence"

        return {
            "inputs": {
                "sequence_data": sequence,
                "omics_data": base64.b64encode(omics.tobytes()).decode("utf-8"),
                "array_data": base64.b64encode(array_input.tobytes()).decode("utf-8"),
                "image_data": _serialize_image(Image.fromarray(image)),
                "tabular_data": {
                    "column1": "Positive",
                    "column2": 0.5,
                },
            },
            "target_labels": {
                "test_output": {"test_target": test_target},
                "output_array": {
                    "output_array": base64.b64encode(array_output.tobytes()).decode("utf-8")
                },
                "output_image": {
                    "output_image": _serialize_image(Image.fromarray(image_output))
                },
                "output_sequence": {"output_sequence": sequence_output},
                "output_survival": {
                    "Event": "0",
                    "Time": 0.5,
                },
            },
            "sample_id": str(uuid.uuid4())
        }

Helper Functions
----------------

Useful functions for data serialization:

.. code-block:: python

    def _serialize_image(image: Image.Image) -> str:
        """Convert PIL Image to base64 string."""
        buffer = io.BytesIO()
        image.save(buffer, format="PNG")
        buffer.seek(0)
        return base64.b64encode(buffer.getvalue()).decode("utf-8")

Dataset Info Structure
----------------------

The server must also provide correct shape information for array-based modalities:

.. code-block:: python

    {
        "inputs": {
            "sequence_data": {"type": "sequence"},
            "tabular_data": {"type": "tabular"},
            "omics_data": {"type": "omics", "shape": [4, 100]},
            "image_data": {"type": "image", "shape": [16, 16, 3]},
            "array_data": {"type": "array", "shape": [10, 5]},
        },
        "outputs": {
            "test_output": {"type": "tabular"},
            "output_array": {"type": "array", "shape": [5, 3]},
            "output_image": {"type": "image", "shape": [16, 16, 3]},
            "output_sequence": {"type": "sequence"},
            "output_survival": {"type": "survival"},
        }
    }


Remember, once your server is properly implemented,
EIR should handle all the client-side functionality.
Users only need to specify the WebSocket URL in their EIR configurations after
it's implemented to start using the streaming data functionality.