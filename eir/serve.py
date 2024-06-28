import argparse

import torchtext
import uvicorn
from fastapi import FastAPI

torchtext.disable_torchtext_deprecation_warning()

from eir.serve_modules.serve_api import create_info_endpoint, create_predict_endpoint
from eir.serve_modules.serve_experiment_io import (
    ServeExperiment,
    load_experiment_for_serve,
)
from eir.utils.logging import get_logger

logger = get_logger(name=__name__, tqdm_compatible=True)

app = FastAPI()


def load_experiment(model_path: str, device: str) -> ServeExperiment:
    serve_experiment = load_experiment_for_serve(
        model_path=model_path,
        device=device,
    )
    return serve_experiment


def get_cl_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Load a model for serving.")
    parser.add_argument(
        "--model-path",
        type=str,
        required=True,
        help="Path to the model file",
    )

    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        help="Device to run the model on",
    )

    parser.add_argument(
        "--port",
        type=int,
        default=8000,
        help="Port to run the server on",
    )

    parser.add_argument(
        "--host",
        type=str,
        default="0.0.0.0",
        help="Host to run the server on",
    )

    args = parser.parse_args()

    return args


def main():
    args = get_cl_args()

    serve_experiment = load_experiment(
        model_path=args.model_path,
        device=args.device,
    )
    logger.info(f"Model loaded from {args.model_path}")

    create_info_endpoint(
        app=app,
        serve_experiment=serve_experiment,
    )

    create_predict_endpoint(
        app=app,
        configs=serve_experiment.configs.input_configs,
        serve_experiment=serve_experiment,
    )

    uvicorn.run(app=app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
