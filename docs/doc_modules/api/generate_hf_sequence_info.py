import inspect
import json
import re
from pathlib import Path

import requests
from transformers import AutoConfig, PretrainedConfig

from eir.setup.setup_utils import get_all_hf_model_names

UPDATE_HF_DOCS = False
CACHE_FILE = "docs/source/_static/sequence_model_overview_cache.json"


def run_all():
    output_file = Path("docs/api/sequence_models.rst")

    header = get_header()

    model_names = get_all_hf_model_names()

    models = retrieve_configurable_models(model_name_list=list(model_names))

    configurable_models_header = _get_configurable_models_header()
    pretrained_models_header = _get_pretrained_models_header()
    configurable_models_rst_string = generate_configurable_model_rst_string(
        models=models
    )

    with open(output_file, "w") as f:
        f.write(header)
        f.write(configurable_models_header)
        f.write(pretrained_models_header)
        f.write(configurable_models_rst_string)


def get_header() -> str:
    header = (
        ".. _external-sequence-models:\n\n"
        "Sequence Models\n"
        "===============\n\n"
        "This page contains the list of external sequence models that can "
        "be used with EIR, coming from the excellent "
        "`Transformers <https://huggingface.co/docs/transformers/index>`__"
        " library.\n\n"
        "There are 3 ways to use these models:\n\n"
        "- Configure and train specific architectures (e.g. BERT with chosen "
        "number of layers) from scratch.\n"
        "- Train a specific architecture (e.g. ``bert-base-uncased``) from scratch.\n"
        "- Use a pre-trained model (e.g. ``bert-base-uncased``) and fine-tune it.\n\n"
        "Please refer to `this page <https://huggingface.co/models>`__ "
        "for a complete list of pre-defined architectures, "
        "with the option of using pre-trained weights.\n\n"
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
        "For example, the ``LongFormer`` architecture includes the "
        "``num_attention_heads`` and "
        "``num_hidden_layers`` parameters, and can be configured as follows:\n\n"
        ".. literalinclude:: ../tutorials/tutorial_files/a_using_eir/"
        "04_pretrained_sequence_tutorial/"
        "04_imdb_input_longformer.yaml"
        "\n    :language: yaml"
        "\n    :caption: input_configurable_sequence_model.yaml\n\n"
    )
    return header


def _get_pretrained_models_header() -> str:
    header = (
        "**Pretrained Models**\n\n"
        "We can also fine-tune or train a specific architecture from scratch. "
        "For example, a ``tiny-bert`` model like so:\n\n"
        ".. literalinclude:: ../tutorials/tutorial_files/a_using_eir/"
        "04_pretrained_sequence_tutorial/"
        "04_imdb_input_tiny-bert.yaml"
        "\n    :language: yaml"
        "\n    :caption: input_pre_trained_sequence_model.yaml\n\n"
        "Below is a list of the configurable models that can be used with EIR.\n\n"
    )
    return header


def load_cache() -> dict:
    if Path(CACHE_FILE).exists() and not UPDATE_HF_DOCS:
        with open(CACHE_FILE) as f:
            return json.load(f)
    else:
        return {}


def save_cache(cache: dict):
    with open(CACHE_FILE, "w") as f:
        json.dump(cache, f)


def retrieve_configurable_models(
    model_name_list: list[str],
) -> list[tuple[str, str]]:
    models_ = []

    for model_name in model_name_list:
        config = AutoConfig.for_model(model_type=model_name)
        if isinstance(config, PretrainedConfig):
            docstring = inspect.getdoc(config.__class__)
            docstring = _add_overview(docstring=docstring, model_type=model_name)

            docstring = markdown_to_rst(md_string=docstring)
            docstring = trim_docstring(docstring=docstring)
            docstring = _parse_docstring(docstring=docstring)

            signature = str(inspect.signature(config.__class__))
            full_import_path = (
                f"{config.__class__.__module__}.{config.__class__.__name__}{signature}"
            )

            models_.append((full_import_path, docstring))

    return models_


def _add_overview(docstring: str, model_type: str) -> str:
    overview = get_model_overview(model_name=model_type)

    if not overview:
        return docstring

    matches = [m.start() for m in re.finditer(r"Args:|Arguments:", docstring)]
    if matches:
        args_start = matches[0]
    else:
        raise ValueError("Could not find args start.")

    overview = overview.replace("## Overview", "").lstrip()
    return overview + docstring[args_start:]


def markdown_to_rst(md_string: str) -> str:
    # Code blocks
    rst_string = re.sub(
        r"```python\n(.*?)\n```", r"::\n\n\1", md_string, flags=re.DOTALL
    )

    # Inline code within brackets
    rst_string = re.sub(r"\[\`(.*?)\`\]", r"``\1``", rst_string)

    # Inline code not within brackets
    rst_string = re.sub(r"`(.*?)`", r"`\1`", rst_string)

    # Links
    rst_string = re.sub(
        r"\[(.*?)\]\((.*?)\)", r"`\1 <\2>`__", rst_string, flags=re.DOTALL
    )

    return rst_string


def trim_docstring(docstring: str) -> str:
    lines = docstring.splitlines()
    for i, line in enumerate(lines):
        if (
            line.strip()
            .lower()
            .startswith(
                (
                    "example:",
                    "examples:",
                    "usage:",
                    "example usage:",
                )
            )
        ):
            return "\n".join(lines[:i]).strip()

    return docstring


def _parse_docstring(docstring: str) -> str:
    return docstring


def generate_configurable_model_rst_string(models: list[tuple[str, str]]) -> str:
    rst_string = ""

    for full_import_path, docstring in models:
        rst_string += f".. class:: {full_import_path}\n"
        rst_string += f"\n{docstring}\n\n"

    return rst_string


def get_model_overview(model_name: str) -> str:
    cache = load_cache()
    if model_name in cache and not UPDATE_HF_DOCS:
        return cache[model_name]

    url = (
        f"https://raw.githubusercontent.com/huggingface/transformers/main/docs/"
        f"source/en/model_doc/{model_name}.md"
    )
    response = requests.get(url)
    content = response.text

    overview_start = content.find("## Overview")
    if overview_start == -1:
        return ""

    content = content[overview_start:]

    next_section = content.find("## ", 11)
    if next_section != -1:
        content = content[:next_section]

    cache[model_name] = content
    save_cache(cache=cache)

    return content
