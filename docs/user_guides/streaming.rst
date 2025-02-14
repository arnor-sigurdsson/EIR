.. _streaming-data-guide:

.. role:: raw-html(raw)
    :format: html

Streaming Data Guide
====================

This guide covers EIR's streaming data functionality,
which allows for real-time data streaming during training.
The guide focuses on how to implement a compatible
WebSocket server that can stream data to EIR.

Overview
--------

EIR includes built-in support
for receiving streaming data via WebSocket connections.
To use this functionality, you only need to:

1. Implement a WebSocket server that follows EIR's protocol specification
2. Point to your server's WebSocket address in EIR's configuration files

For example,
to use streaming data in EIR,
you would simply specify the WebSocket URL in your configuration:

.. code-block:: yaml

    output_info:
      output_source: ws://localhost:8000/ws
      output_name: text_output
      output_type: sequence

EIR will automatically handle the connection, data receiving, and processing.

Protocol Specification
----------------------

To be compatible with EIR, your WebSocket server must implement the following protocol:

Message Structure
^^^^^^^^^^^^^^^^^

All messages use JSON format:

.. code-block:: python

    {
        "type": str,    # Message type
        "payload": Any  # Message payload
    }

Required Message Types
^^^^^^^^^^^^^^^^^^^^^^

1. ``handshake``
   - First message exchanged after connection
   - Server must verify protocol version

2. ``getInfo``
   - Returns dataset information including shapes and types
   - Required for EIR to set up proper data processing

3. ``getData``
   - Returns batches of data samples
   - Must handle batch_size parameter

Example Server Implementation
-----------------------------

Here's a minimal example of a compatible server using FastAPI:

.. code-block:: python

    from fastapi import FastAPI, WebSocket
    import numpy as np
    import base64

    app = FastAPI()
    PROTOCOL_VERSION = "1.0"  # Must match EIR's version

    class ConnectionManager:
        async def connect(self, websocket: WebSocket):
            await websocket.accept()
            handshake = await websocket.receive_json()
            if handshake["type"] != "handshake" or handshake["version"] != PROTOCOL_VERSION:
                await websocket.close()
                return False
            return True

    manager = ConnectionManager()

    @app.websocket("/ws")
    async def websocket_endpoint(websocket: WebSocket):
        if not await manager.connect(websocket):
            return

        try:
            while True:
                data = await websocket.receive_json()

                if data["type"] == "getInfo":
                    # Example: Define your dataset structure
                    # See below in the tutorial for more details on what to include
                    # for different modalities
                    info = {
                        "inputs": {
                            "sequence_data": {
                                "type": "sequence"
                            }
                        },
                        "outputs": {
                            "text_output": {
                                "type": "sequence"
                            }
                        }
                    }
                    await websocket.send_json({
                        "type": "info",
                        "payload": info
                    })

                elif data["type"] == "getData":
                    batch_size = data["payload"]["batch_size"]
                    # Example: Generate/fetch your data
                    # See below in the tutorial for more details on what structure to use
                    # for different modalities
                    batch = [
                        {
                            "inputs": {"sequence_data": "example input"},
                            "target_labels": {"text_output": "example output"},
                            "sample_id": "unique_id"
                        }
                        for _ in range(batch_size)
                    ]

                    await websocket.send_json({
                        "type": "data",
                        "payload": batch
                    })

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