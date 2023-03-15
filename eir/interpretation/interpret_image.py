from pathlib import Path
from typing import TYPE_CHECKING, Sequence

import matplotlib.pyplot as plt
import numpy as np
import torch
from aislib.misc_utils import ensure_path_exists
from captum.attr._utils.visualization import (
    visualize_image_attr_multiple,
)
from matplotlib.colors import LinearSegmentedColormap
from torchvision.transforms import Normalize

from eir.interpretation.interpretation_utils import (
    get_target_class_name,
    get_basic_sample_attributions_to_analyse_generator,
)

if TYPE_CHECKING:
    from eir.train import Experiment
    from eir.interpretation.interpretation import SampleAttribution
    from eir.setup.input_setup import ImageNormalizationStats


def analyze_image_input_attributions(
    experiment: "Experiment",
    input_name: str,
    target_column_name: str,
    output_name: str,
    target_column_type: str,
    attribution_outfolder: Path,
    all_attributions: Sequence["SampleAttribution"],
) -> None:
    exp = experiment

    output_object = exp.outputs[output_name]
    target_transformer = output_object.target_transformers[target_column_name]

    input_object = exp.inputs[input_name]
    interpretation_config = input_object.input_config.interpretation_config

    samples_to_act_analyze_gen = get_basic_sample_attributions_to_analyse_generator(
        interpretation_config=interpretation_config, all_attributions=all_attributions
    )

    for sample_attribution in samples_to_act_analyze_gen:
        sample_target_labels = sample_attribution.sample_info.target_labels

        cur_label_name = get_target_class_name(
            sample_label=sample_target_labels[output_name][target_column_name],
            target_transformer=target_transformer,
            column_type=target_column_type,
            target_column_name=target_column_name,
        )

        attributions = sample_attribution.sample_attributions[input_name].squeeze()
        raw_input = sample_attribution.raw_inputs[input_name].cpu().numpy().squeeze()
        attributions = attributions.transpose(1, 2, 0)
        raw_input = un_normalize(
            normalized_img=raw_input,
            normalization_stats=input_object.normalization_stats,
        )
        raw_input = raw_input.transpose(1, 2, 0)

        cmap = get_default_img_attribution_cmap()
        figure, _ = visualize_image_attr_multiple(
            attr=attributions,
            original_image=raw_input,
            methods=["original_image", "heat_map"],
            signs=["all", "absolute_value"],
            show_colorbar=True,
            cmap=cmap,
        )

        name = f"image_{sample_attribution.sample_info.ids[0]}_{cur_label_name}.pdf"
        outpath = attribution_outfolder / "single_samples" / name
        ensure_path_exists(path=outpath)
        figure.savefig(outpath, dpi=300)
        plt.close("all")


def un_normalize(
    normalized_img: np.ndarray, normalization_stats: "ImageNormalizationStats"
):
    """
    Clip because sometimes we get values like 1.0000001 which will cause some libraries
    to do =/ 255, resulting in a black image.
    """
    means = torch.Tensor(normalization_stats.channel_means)
    stds = torch.Tensor(normalization_stats.channel_stds)
    unnorm = Normalize(
        mean=(-means / stds).tolist(),
        std=(1.0 / stds).tolist(),
    )

    img_as_tensor = torch.Tensor(normalized_img)
    img_unnormalized = unnorm(img_as_tensor).cpu().numpy()
    img_final = img_unnormalized.clip(None, 1.0)

    return img_final


def get_default_img_attribution_cmap() -> LinearSegmentedColormap:
    default_cmap = LinearSegmentedColormap.from_list(
        "custom blue", [(0, "#ffffff"), (0.25, "#252b36"), (1, "#000000")], N=256
    )

    return default_cmap
