from pathlib import Path
from typing import TYPE_CHECKING, Sequence

import matplotlib.pyplot as plt
import numpy as np
import shap
import torch
from aislib.misc_utils import ensure_path_exists
from torchvision.transforms import Normalize

from eir.interpretation.interpretation_utils import (
    get_target_class_name,
    get_basic_sample_activations_to_analyse_generator,
)

if TYPE_CHECKING:
    from eir.train import Experiment
    from eir.interpretation.interpretation import SampleActivation
    from eir.setup.input_setup import ImageNormalizationStats


def analyze_image_input_activations(
    experiment: "Experiment",
    input_name: str,
    target_column_name: str,
    output_name: str,
    target_column_type: str,
    activation_outfolder: Path,
    all_activations: Sequence["SampleActivation"],
) -> None:

    exp = experiment

    output_object = exp.outputs[output_name]
    target_transformer = output_object.target_transformers[target_column_name]

    input_object = exp.inputs[input_name]
    interpretation_config = input_object.input_config.interpretation_config

    samples_to_act_analyze_gen = get_basic_sample_activations_to_analyse_generator(
        interpretation_config=interpretation_config, all_activations=all_activations
    )

    for sample_activation in samples_to_act_analyze_gen:

        sample_target_labels = sample_activation.sample_info.target_labels

        cur_label_name = get_target_class_name(
            sample_label=sample_target_labels[output_name][target_column_name],
            target_transformer=target_transformer,
            column_type=target_column_type,
            target_column_name=target_column_name,
        )

        shap_values = sample_activation.sample_activations[input_name].squeeze()
        raw_input = sample_activation.raw_inputs[input_name].cpu().numpy().squeeze()
        shap_values = shap_values.transpose(1, 2, 0)
        raw_input = unnormalize(
            normalized_img=raw_input,
            normalization_stats=input_object.normalization_stats,
        )
        raw_input = raw_input.transpose(1, 2, 0)

        shap.image_plot(
            shap_values=shap_values,
            pixel_values=raw_input,
            show=False,
        )
        cur_figure = plt.gcf()

        outpath = (
            activation_outfolder
            / "single_samples"
            / f"image_{sample_activation.sample_info.ids[0]}_{cur_label_name}.pdf"
        )
        ensure_path_exists(path=outpath)
        cur_figure.savefig(outpath, dpi=300)
        plt.close("all")


def unnormalize(
    normalized_img: np.ndarray, normalization_stats: "ImageNormalizationStats"
):
    """
    Clip because sometimes we get values like 1.0000001 which will cause shap.image
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
