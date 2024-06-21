from collections import defaultdict
from copy import copy
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Dict, Iterable, Mapping, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
import torchtext

torchtext.disable_torchtext_deprecation_warning()

import torchtext.vocab
from aislib.misc_utils import ensure_path_exists
from captum.attr._utils.visualization import (
    _get_color,
    format_classname,
    format_word_importances,
)
from torchtext.vocab import Vocab

from eir.interpretation.interpretation_utils import (
    get_basic_sample_attributions_to_analyse_generator,
    get_long_format_attribution_df,
    get_target_class_name,
    plot_attributions_bar,
    stratify_attributions_by_target_classes,
)
from eir.setup.input_setup_modules.setup_sequence import ComputedSequenceInputInfo
from eir.setup.output_setup_modules.tabular_output_setup import (
    ComputedTabularOutputInfo,
)
from eir.setup.schemas import BasicInterpretationConfig
from eir.utils.logging import get_logger

if TYPE_CHECKING:
    from eir.interpretation.interpretation import SampleAttribution
    from eir.train import Experiment

logger = get_logger(name=__name__)


@dataclass()
class SequenceVisualizationDataRecord:
    sample_id: str
    token_attributions: np.ndarray
    label_name: str
    attribution_score: float
    raw_input_tokens: Sequence[str]


def analyze_sequence_input_attributions(
    experiment: "Experiment",
    input_name: str,
    output_name: str,
    target_column_name: str,
    target_column_type: str,
    attribution_outfolder: Path,
    all_attributions: list["SampleAttribution"],
    expected_target_classes_attributions: np.ndarray,
) -> None:
    exp = experiment

    output_object = exp.outputs[output_name]
    assert isinstance(output_object, ComputedTabularOutputInfo)
    target_transformer = output_object.target_transformers[target_column_name]

    input_object = exp.inputs[input_name]
    assert isinstance(input_object, ComputedSequenceInputInfo)
    vocab = input_object.vocab

    interpretation_config = input_object.input_config.interpretation_config
    assert isinstance(interpretation_config, BasicInterpretationConfig)

    samples_to_act_analyze_gen = get_basic_sample_attributions_to_analyse_generator(
        interpretation_config=interpretation_config, all_attributions=all_attributions
    )

    viz_records = []

    for sample_attribution in samples_to_act_analyze_gen:
        sample_target_labels = sample_attribution.sample_info.target_labels

        cur_label_name = get_target_class_name(
            sample_label=sample_target_labels[output_name][target_column_name],
            target_transformer=target_transformer,
            column_type=target_column_type,
            target_column_name=target_column_name,
        )

        extracted_sample_info = extract_sample_info_for_sequence_attribution(
            sample_attribution_object=sample_attribution,
            cur_label_name=cur_label_name,
            output_name=output_name,
            target_column_name=target_column_name,
            target_column_type=target_column_type,
            input_name=input_name,
            vocab=vocab,
            expected_target_classes_attributions=expected_target_classes_attributions,
        )

        index_to_truncate = get_sequence_index_to_truncate_unknown(
            raw_inputs=extracted_sample_info.raw_inputs
        )

        truncated_sample_info = truncate_sequence_attribution_to_padding(
            sequence_attribution_sample_info=extracted_sample_info,
            truncate_start_idx=index_to_truncate,
        )
        tsi = truncated_sample_info

        if not tsi.raw_inputs:
            logger.debug(
                "Skipping sequence attribution analysis of single sample %s as it is "
                "empty after truncating unknowns.",
                sample_attribution.sample_info.ids[0],
            )
            continue

        viz_record = SequenceVisualizationDataRecord(
            sample_id=sample_attribution.sample_info.ids[0],
            token_attributions=tsi.sequence_attributions,
            label_name=tsi.sample_target_label_name,
            attribution_score=tsi.sequence_attributions.sum(),
            raw_input_tokens=tsi.raw_inputs,
        )
        viz_records.append(viz_record)

    html_string = get_sequence_html(data_records=viz_records)
    outpath = attribution_outfolder / "single_samples.html"
    ensure_path_exists(path=outpath)
    save_html(out_path=outpath, html_string=html_string)

    acts_stratified_by_target = stratify_attributions_by_target_classes(
        all_attributions=all_attributions,
        target_transformer=target_transformer,
        output_name=output_name,
        target_column=target_column_name,
        column_type=target_column_type,
    )

    for class_name, target_attributions in acts_stratified_by_target.items():
        token_importances = get_sequence_token_importance(
            attributions=target_attributions, vocab=vocab, input_name=input_name
        )
        df_token_importances = get_long_format_attribution_df(
            parsed_attributions=token_importances
        )
        target_outfolder = attribution_outfolder / f"{class_name}"
        ensure_path_exists(path=target_outfolder, is_folder=True)
        plot_attributions_bar(
            df_attributions=df_token_importances,
            output_path=target_outfolder / f"token_influence_{class_name}.pdf",
            title=f"{target_column_name} – {class_name}",
            use_bootstrap=True,
        )
        df_token_importances.to_csv(
            path_or_buf=target_outfolder / f"token_influence_{class_name}.csv"
        )


@dataclass
class SequenceAttributionSampleInfo:
    sequence_attributions: np.ndarray
    raw_inputs: Sequence[str]
    expected_attr_value: float
    sample_target_label_name: str


def extract_sample_info_for_sequence_attribution(
    sample_attribution_object: "SampleAttribution",
    cur_label_name: str,
    output_name: str,
    target_column_name: str,
    target_column_type: str,
    input_name: str,
    vocab: Vocab,
    expected_target_classes_attributions: np.ndarray,
) -> SequenceAttributionSampleInfo:
    attributions = sample_attribution_object.sample_attributions[input_name]

    sample_tokens = sample_attribution_object.raw_inputs[input_name]
    raw_inputs = extract_raw_inputs_from_tokens(tokens=sample_tokens, vocab=vocab)

    cur_sample_expected_value = _parse_out_sequence_expected_value(
        sample_target_labels=sample_attribution_object.sample_info.target_labels,
        output_name=output_name,
        target_column_name=target_column_name,
        target_column_type=target_column_type,
        expected_values=expected_target_classes_attributions,
    )

    extracted_sequence_info = SequenceAttributionSampleInfo(
        sequence_attributions=attributions,
        raw_inputs=raw_inputs,
        expected_attr_value=cur_sample_expected_value,
        sample_target_label_name=cur_label_name,
    )

    return extracted_sequence_info


def extract_raw_inputs_from_tokens(
    tokens: torch.Tensor, vocab: torchtext.vocab.Vocab
) -> Sequence[str]:
    raw_inputs = vocab.lookup_tokens(tokens.squeeze().tolist())
    return raw_inputs


def _parse_out_sequence_expected_value(
    sample_target_labels: Dict[str, Dict[str, torch.Tensor]],
    output_name: str,
    target_column_name: str,
    expected_values: np.ndarray,
    target_column_type: str,
) -> float:
    if target_column_type == "con":
        assert len(expected_values) == 1
        return expected_values[0]

    cur_base_values_index = sample_target_labels[output_name][target_column_name].item()
    assert isinstance(cur_base_values_index, int)
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


def truncate_sequence_attribution_to_padding(
    sequence_attribution_sample_info: SequenceAttributionSampleInfo,
    truncate_start_idx: int,
) -> SequenceAttributionSampleInfo:
    si = sequence_attribution_sample_info

    attrs_truncated, raw_inputs_truncated = _truncate_attributions_and_raw_inputs(
        attributions=si.sequence_attributions,
        raw_inputs=si.raw_inputs,
        truncate_start_idx=truncate_start_idx,
    )

    truncated_attribution = SequenceAttributionSampleInfo(
        sequence_attributions=attrs_truncated,
        raw_inputs=raw_inputs_truncated,
        expected_attr_value=si.expected_attr_value,
        sample_target_label_name=si.sample_target_label_name,
    )

    return truncated_attribution


def _truncate_attributions_and_raw_inputs(
    attributions: np.ndarray, raw_inputs: Sequence[str], truncate_start_idx: int
) -> Tuple[np.ndarray, Sequence[str]]:
    attrs_copy = copy(attributions)
    raw_inputs_copy = copy(raw_inputs)

    attrs_summed = attrs_copy.squeeze().sum(1)
    attrs_truncated = attrs_summed[:truncate_start_idx]

    raw_inputs_truncated = [i + " " for i in raw_inputs_copy][:truncate_start_idx]

    return attrs_truncated, raw_inputs_truncated


def save_html(out_path: Path, html_string: str) -> None:
    with open(out_path, "w", encoding="utf-8") as outfile:
        outfile.write(html_string)


def get_sequence_token_importance(
    attributions: Sequence["SampleAttribution"], vocab: Vocab, input_name: str
) -> Dict[str, list[float]]:
    token_importances = defaultdict(list)

    for act in attributions:
        orig_seq_input = extract_raw_inputs_from_tokens(
            tokens=act.raw_inputs[input_name], vocab=vocab
        )
        seq_attrs = act.sample_attributions[input_name]
        seq_attrs_abs = seq_attrs.squeeze()
        assert len(seq_attrs_abs.shape) == 2

        seq_attrs_values_sum = seq_attrs_abs.sum(1).squeeze()
        assert len(seq_attrs_values_sum) == len(orig_seq_input)

        for token, attribution in zip(orig_seq_input, seq_attrs_values_sum):
            token_importances[token].append(attribution)

    return token_importances


def get_sequence_feature_importance_df(
    token_importances: dict[str, list[float]]
) -> pd.DataFrame:

    series_dict: Mapping[str, pd.Series] = {
        k: pd.Series(v) for k, v in token_importances.items()
    }

    df: pd.DataFrame = pd.concat(series_dict)
    df = df.reset_index(level=0).reset_index(drop=True)

    df = df.rename(columns={df.columns[0]: "Input", df.columns[1]: "Attribution"})

    return df


def get_sequence_html(
    data_records: Iterable[SequenceVisualizationDataRecord], legend: bool = True
) -> str:
    dom = [
        "<table width: 100%; border: 2px solid black; border-collapse: collapse;>",
        '<table class="table" style="text-align: center;">',
    ]
    rows = [
        "<tr><th>ID</th>"
        "<th>True Label</th>"
        "<th>Attribution Score</th>"
        "<th>Token Importance</th>"
    ]
    for record in data_records:
        rows.append(
            "".join(
                [
                    "<tr>",
                    format_classname(record.sample_id),
                    format_classname(record.label_name),
                    format_classname("{0:.2f}".format(record.attribution_score)),
                    format_word_importances(
                        record.raw_input_tokens, record.token_attributions
                    ),
                    "<tr>",
                ]
            )
        )

    if legend:
        dom.append(
            '<div style="border-top: 1px solid; margin-top: 5px; \
            padding-top: 5px; display: inline-block">'
        )
        dom.append("<b>Legend: </b>")

        for value, label in zip([-1, 0, 1], ["Negative", "Neutral", "Positive"]):
            dom.append(
                '<span style="display: inline-block; width: 10px; height: 10px; \
                border: 1px solid; background-color: \
                {value}"></span> {label}  '.format(
                    value=_get_color(value), label=label
                )
            )
        dom.append("</div>")

    dom.append("".join(rows))
    dom.append("</table>")
    html = "".join(dom)

    return html
