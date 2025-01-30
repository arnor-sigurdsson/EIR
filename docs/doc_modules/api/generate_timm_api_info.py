import importlib
import inspect
import pkgutil
from pathlib import Path
from typing import Any


def run_all():
    output_file = Path("docs/api/image_models.rst")

    header = get_header()

    models = retrieve_configurable_models("timm.models")

    configurable_models_header = _get_configurable_models_header()
    configurable_models_rst_string = generate_configurable_model_rst_string(
        models=models
    )

    with open(output_file, "w") as f:
        f.write(header)
        f.write(configurable_models_header)
        f.write(configurable_models_rst_string)


def get_header() -> str:
    header = (
        ".. _external-image-models:\n\n"
        "Image Models\n"
        "============\n\n"
        "This page contains the list of external image models that can "
        "be used with EIR, coming from the great "
        "`timm <https://huggingface.co/docs/timm>`__ library.\n\n"
        "There are 3 ways to use these models:\n\n"
        "- Configure and train specific architectures (e.g. ResNet with chosen "
        "number of layers) from scratch.\n"
        "- Train a specific architecture (e.g. ``resnet18``) from scratch.\n"
        "- Use a pre-trained model (e.g. ``resnet18``) and fine-tune it.\n\n"
        "Please refer to `this page <https://huggingface.co/docs/timm/models>`__ "
        "for more detailed information about configurable architectures, "
        "and `this page <https://huggingface.co/timm>`__ for a list of "
        "pre-defined architectures, with the option of using pre-trained weights.\n\n"
    )

    return header


def _get_configurable_models_header() -> str:
    header = (
        "Configurable Models\n"
        "-------------------\n\n"
        "The following models can be configured and trained from scratch.\n\n"
        "The model type is specified in the ``model_type`` field of the "
        "configuration, while the model specific configuration is specified "
        "in the ``model_init_config`` field.\n\n"
        "For example, the ``ResNet`` architecture includes the ``layers`` and "
        "``block`` parameters, and can be configured as follows:\n\n"
        ".. literalinclude:: ../tutorials/tutorial_files/a_using_eir/05_image_tutorial/"
        "inputs.yaml"
        "\n    :language: yaml"
        "\n    :caption: input_configurable_image_model.yaml\n\n"
    )

    return header


def retrieve_configurable_models(
    package: str, block_suffix: str = "Block"
) -> list[tuple[Any, str | None, list[str]]]:
    models_ = []

    package_iter = pkgutil.iter_modules(importlib.import_module(package).__path__)
    for _importer, modname, is_pkg in package_iter:
        if not is_pkg and not modname.startswith("_") and modname != "registry":
            module = importlib.import_module(f"{package}.{modname}")

            if hasattr(module, "__all__"):
                model_name = module.__all__[0]

                if not model_name.startswith("_"):
                    model_class = getattr(module, model_name, None)
                    is_class = inspect.isclass(model_class)
                    is_block = model_name.endswith(block_suffix)

                    if is_class and not is_block:
                        docstring = inspect.getdoc(model_class)
                        parameters = list(
                            inspect.signature(model_class).parameters.keys()
                        )

                        full_import_path = f"{module.__name__}.{model_name}"

                        models_.append((full_import_path, docstring, parameters))

    return models_


def generate_configurable_model_rst_string(
    models: list[tuple[Any, str | None, list[str]]],
) -> str:
    rst_string = ""

    for full_import_path, _docstring, _parameters in models:
        rst_string += f".. autoclass:: {full_import_path}\n"
        rst_string += "   :members:\n"
        rst_string += "   :exclude-members: forward\n\n"

    return rst_string
