import io
from collections import defaultdict
from copy import copy
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Sequence, Literal, Tuple, Dict

import numpy as np
import pandas as pd
import shap
import torch
from aislib.misc_utils import ensure_path_exists, get_logger
from shap._explanation import Explanation
from torchtext.vocab import Vocab

from eir.interpretation.interpretation_utils import (
    get_target_class_name,
    stratify_activations_by_target_classes,
    plot_activations_bar,
    get_basic_sample_activations_to_analyse_generator,
)
from eir.visualization.sequence_visualization_forward_port import text

if TYPE_CHECKING:
    from eir.train import Experiment
    from eir.interpretation.interpretation import SampleActivation

logger = get_logger(name=__name__)


def analyze_sequence_input_activations(
    experiment: "Experiment",
    input_name: str,
    output_name: str,
    target_column_name: str,
    target_column_type: str,
    activation_outfolder: Path,
    all_activations: Sequence["SampleActivation"],
    expected_target_classes_shap_values: Sequence[float],
) -> None:

    exp = experiment

    output_object = exp.outputs[output_name]
    target_transformer = output_object.target_transformers[target_column_name]

    input_object = exp.inputs[input_name]
    interpretation_config = input_object.input_config.interpretation_config

    samples_to_act_analyze_gen = get_basic_sample_activations_to_analyse_generator(
        interpretation_config=interpretation_config, all_activations=all_activations
    )
    vocab = exp.inputs[input_name].vocab

    for sample_activation in samples_to_act_analyze_gen:

        sample_target_labels = sample_activation.sample_info.target_labels

        cur_label_name = get_target_class_name(
            sample_label=sample_target_labels[output_name][target_column_name],
            target_transformer=target_transformer,
            column_type=target_column_type,
            target_column_name=target_column_name,
        )

        extracted_sample_info = extract_sample_info_for_sequence_activation(
            sample_activation_object=sample_activation,
            cur_label_name=cur_label_name,
            output_name=output_name,
            target_column_name=target_column_name,
            target_column_type=target_column_type,
            input_name=input_name,
            vocab=vocab,
            expected_target_classes_shap_values=expected_target_classes_shap_values,
        )

        index_to_truncate = get_sequence_index_to_truncate_unknown(
            raw_inputs=extracted_sample_info.raw_inputs
        )

        truncated_sample_info = truncate_sequence_activation_to_padding(
            sequence_activation_sample_info=extracted_sample_info,
            truncate_start_idx=index_to_truncate,
        )

        if not truncated_sample_info.raw_inputs:
            logger.debug(
                "Skipping sequence activation analysis of single sample %s as it is "
                "empty after truncating unknowns.",
                sample_activation.sample_info.ids[0],
            )
            continue

        explanation = Explanation(
            values=truncated_sample_info.sequence_shap_values,
            data=np.array(truncated_sample_info.raw_inputs),
            base_values=extracted_sample_info.expected_shap_value,
        )

        html_string = text(shap_values=explanation, display=False)

        outpath = (
            activation_outfolder
            / "single_samples"
            / f"sequence_{sample_activation.sample_info.ids[0]}_{cur_label_name}.html"
        )
        ensure_path_exists(path=outpath)
        save_html(out_path=outpath, html_string=html_string)

    acts_stratified_by_target = stratify_activations_by_target_classes(
        all_activations=all_activations,
        target_transformer=target_transformer,
        output_name=output_name,
        target_column=target_column_name,
        column_type=target_column_type,
    )

    for class_name, target_activations in acts_stratified_by_target.items():
        token_importances = get_sequence_token_importance(
            activations=target_activations, vocab=vocab, input_name=input_name
        )
        df_token_importances = get_sequence_feature_importance_df(
            token_importances=token_importances
        )
        target_outfolder = activation_outfolder / f"{class_name}"
        ensure_path_exists(path=target_outfolder, is_folder=True)
        plot_activations_bar(
            df_activations=df_token_importances,
            outpath=target_outfolder / f"token_influence_{class_name}.pdf",
            title=f"{target_column_name} – {class_name}",
        )
        df_token_importances.to_csv(
            path_or_buf=target_outfolder / f"token_influence_{class_name}.csv"
        )


@dataclass
class SequenceActivationSampleInfo:
    sequence_shap_values: np.ndarray
    raw_inputs: Sequence[str]
    expected_shap_value: float
    sample_target_label_name: str


def extract_sample_info_for_sequence_activation(
    sample_activation_object: "SampleActivation",
    cur_label_name: str,
    output_name: str,
    target_column_name: str,
    target_column_type: str,
    input_name: str,
    vocab: Vocab,
    expected_target_classes_shap_values: Sequence[float],
) -> SequenceActivationSampleInfo:

    shap_values = sample_activation_object.sample_activations[input_name]

    sample_tokens = sample_activation_object.raw_inputs[input_name]
    raw_inputs = extract_raw_inputs_from_tokens(tokens=sample_tokens, vocab=vocab)

    cur_sample_expected_value = _parse_out_sequence_expected_value(
        sample_target_labels=sample_activation_object.sample_info.target_labels,
        output_name=output_name,
        target_column_name=target_column_name,
        target_column_type=target_column_type,
        expected_values=expected_target_classes_shap_values,
    )

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


def _parse_out_sequence_expected_value(
    sample_target_labels: Dict[str, torch.Tensor],
    output_name: str,
    target_column_name: str,
    expected_values: Sequence[float],
    target_column_type: str,
) -> float:

    if target_column_type == "con":
        assert len(expected_values) == 1
        return expected_values[0]

    cur_base_values_index = sample_target_labels[output_name][target_column_name].item()
    cur_sample_expected_value = expected_values[cur_base_values_index]

    return cur_sample_expected_value


def get_sequence_index_to_truncate_unknown(
    raw_inputs: Sequence[str], padding_value: str = "<pad>"
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


def get_sequence_token_importance(
    activations: Sequence["SampleActivation"], vocab: Vocab, input_name: str
) -> Dict[str, float]:

    token_importances = defaultdict(lambda: 0.0)
    token_counts = defaultdict(lambda: 0)

    for act in activations:
        orig_seq_input = extract_raw_inputs_from_tokens(
            tokens=act.raw_inputs[input_name], vocab=vocab
        )
        seq_shap_values = act.sample_activations[input_name]
        seq_shap_values_abs = seq_shap_values.squeeze()
        assert len(seq_shap_values_abs.shape) == 2

        seq_shap_values_sum = seq_shap_values_abs.sum(1).squeeze()
        assert len(seq_shap_values_sum) == len(orig_seq_input)

        for token, shap_value in zip(orig_seq_input, seq_shap_values_sum):
            token_importances[token] += shap_value
            token_counts[token] += 1

    n_samples = len(activations)
    for token, total_shap_value in token_importances.items():
        token_importances[token] = total_shap_value / (token_counts[token] * n_samples)

    return token_importances


def get_sequence_feature_importance_df(
    token_importances: Dict[str, float]
) -> pd.DataFrame:
    df = pd.DataFrame.from_dict(
        token_importances, columns=["Shap_Value"], orient="index"
    )

    df.index.name = "Token"

    return df
