import io
import random
from copy import copy
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Sequence, Literal, Generator, Tuple

import numpy as np
import shap
import torch
from shap._explanation import Explanation
from torchtext.vocab import Vocab

from eir.interpretation.interpretation_utils import get_target_class_name
from eir.setup import schemas

if TYPE_CHECKING:
    from eir.train import Experiment
    from eir.interpretation.interpretation import SampleActivation


def analyze_sequence_input_activations(
    experiment: "Experiment",
    input_name: str,
    target_column_name: str,
    target_column_type: str,
    activation_outfolder: Path,
    all_activations: Sequence["SampleActivation"],
    expected_target_classes_shap_values: Sequence[float],
) -> None:

    exp = experiment

    target_transformer = exp.target_transformers[target_column_name]
    input_object = exp.inputs[input_name]
    interpretation_config = input_object.input_config.interpretation_config

    samples_to_act_analyze_gen = get_sequence_sample_activations_to_analyse_generator(
        interpretation_config=interpretation_config, all_activations=all_activations
    )

    for sample_activation in samples_to_act_analyze_gen:

        sample_target_labels = sample_activation.sample_info.target_labels

        cur_label_name = get_target_class_name(
            sample_label=sample_target_labels[target_column_name],
            target_transformer=target_transformer,
            column_type=target_column_type,
            target_column_name=target_column_name,
        )

        extracted_sample_info = extract_sample_info_for_sequence_activation(
            sample_activation_object=sample_activation,
            cur_label_name=cur_label_name,
            target_column_name=target_column_name,
            input_name=input_name,
            vocab=exp.inputs[input_name].vocab,
            expected_target_classes_shap_values=expected_target_classes_shap_values,
        )

        index_to_truncate = get_sequence_index_to_truncate_unknown(
            raw_inputs=extracted_sample_info.raw_inputs
        )

        truncated_sample_info = truncate_sequence_activation_to_padding(
            sequence_activation_sample_info=extracted_sample_info,
            truncate_start_idx=index_to_truncate,
        )

        explanation = Explanation(
            values=truncated_sample_info.sequence_shap_values,
            data=np.array(truncated_sample_info.raw_inputs),
            base_values=extracted_sample_info.expected_shap_value,
        )

        html_string = shap.plots.text(explanation, display=False)

        outpath = (
            activation_outfolder
            / f"sequence_{sample_activation.sample_info.ids[0]}_{cur_label_name}.html"
        )
        save_html(out_path=outpath, html_string=html_string)


def get_sequence_sample_activations_to_analyse_generator(
    interpretation_config: schemas.SequenceInterpretationConfig,
    all_activations: Sequence["SampleActivation"],
) -> Generator["SampleActivation", None, None]:

    strategy = interpretation_config.interpretation_sampling_strategy
    n_samples = interpretation_config.num_samples_to_interpret

    if strategy == "first_n":
        base = all_activations[:n_samples]
    elif strategy == "random_sample":
        base = random.sample(all_activations, n_samples)
    else:
        raise ValueError()

    manual_samples = interpretation_config.manual_samples_to_interpret
    if manual_samples:
        for activation in all_activations:
            if activation.sample_info.ids in manual_samples:
                base.append(activation)

    for item in base:
        yield item


@dataclass
class SequenceActivationSampleInfo:
    sequence_shap_values: np.ndarray
    raw_inputs: Sequence[str]
    expected_shap_value: float
    sample_target_label_name: str


def extract_sample_info_for_sequence_activation(
    sample_activation_object: "SampleActivation",
    cur_label_name: str,
    target_column_name: str,
    input_name: str,
    vocab: Vocab,
    expected_target_classes_shap_values: Sequence[float],
) -> SequenceActivationSampleInfo:

    shap_values = sample_activation_object.sample_activations[input_name]

    sample_tokens = sample_activation_object.raw_inputs[input_name]
    raw_inputs = extract_raw_inputs_from_tokens(tokens=sample_tokens, vocab=vocab)

    sample_target_labels = sample_activation_object.sample_info.target_labels
    cur_base_values_index = sample_target_labels[target_column_name].item()
    cur_sample_expected_value = expected_target_classes_shap_values[
        cur_base_values_index
    ]

    extracted_sequence_info = SequenceActivationSampleInfo(
        sequence_shap_values=shap_values,
        raw_inputs=raw_inputs,
        expected_shap_value=cur_sample_expected_value,
        sample_target_label_name=cur_label_name,
    )

    return extracted_sequence_info


def extract_raw_inputs_from_tokens(tokens: torch.Tensor, vocab) -> Sequence[str]:
    raw_inputs = vocab.lookup_tokens(tokens.squeeze().tolist())
    return raw_inputs


def get_sequence_index_to_truncate_unknown(
    raw_inputs: Sequence[str], padding_value: str = "<unk>"
) -> int:
    raw_inputs_reversed = raw_inputs[::-1]
    counter = 0
    for element in raw_inputs_reversed:
        if element == padding_value:
            counter += 1
        else:
            break

    index_to_truncate = len(raw_inputs) - counter

    return index_to_truncate


def truncate_sequence_activation_to_padding(
    sequence_activation_sample_info: SequenceActivationSampleInfo,
    truncate_start_idx: int,
) -> SequenceActivationSampleInfo:
    si = sequence_activation_sample_info

    shap_values_truncated, raw_inputs_truncated = _truncate_shap_values_and_raw_inputs(
        shap_values=si.sequence_shap_values,
        raw_inputs=si.raw_inputs,
        truncate_start_idx=truncate_start_idx,
    )

    truncated_activation = SequenceActivationSampleInfo(
        sequence_shap_values=shap_values_truncated,
        raw_inputs=raw_inputs_truncated,
        expected_shap_value=si.expected_shap_value,
        sample_target_label_name=si.sample_target_label_name,
    )

    return truncated_activation


def _truncate_shap_values_and_raw_inputs(
    shap_values: np.ndarray, raw_inputs: Sequence[str], truncate_start_idx: int
) -> Tuple[np.ndarray, Sequence[str]]:

    shap_values_copy = copy(shap_values)
    raw_inputs_copy = copy(raw_inputs)

    shap_values_summed = shap_values_copy.squeeze().sum(1)
    shap_values_truncated = shap_values_summed[:truncate_start_idx]

    raw_inputs_truncated = [i + " " for i in raw_inputs_copy][:truncate_start_idx]

    return shap_values_truncated, raw_inputs_truncated


def save_html(out_path: Path, html_string: str) -> None:

    with open(out_path, "w", encoding="utf-8") as outfile:

        outfile.write("<html><head><script>\n")

        shap_plots_module_path = Path(shap.plots.__file__).parent
        bundle_path = shap_plots_module_path / "resources" / "bundle.js"

        with io.open(bundle_path, encoding="utf-8") as f:
            bundle_data = f.read()

        outfile.write(bundle_data)
        outfile.write("</script></head><body>\n")

        outfile.write(html_string)

        outfile.write("</body></html>\n")


def get_label_transformer_mapping(
    transformer, order: Literal["int-to-string", "string-to-int"]
) -> dict:

    values = transformer.classes_, transformer.transform(transformer.classes_)
    if order == "int-to-string":
        values = transformer.transform(transformer.classes_), transformer.classes_

    mapping = dict(zip(*values))

    return mapping
