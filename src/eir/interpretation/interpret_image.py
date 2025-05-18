from collections.abc import Sequence
from pathlib import Path
from typing import TYPE_CHECKING

import matplotlib.pyplot as plt
import numpy as np
import torch
from aislib.misc_utils import ensure_path_exists
from captum.attr._utils.visualization import Union, visualize_image_attr_multiple
from torchvision.transforms import Normalize

from eir.interpretation.interpretation_utils import (
    TargetTypeInfo,
    get_appropriate_target_transformer,
    get_basic_sample_attributions_to_analyse_generator,
    get_target_class_name,
)
from eir.setup.input_setup_modules.setup_image import ComputedImageInputInfo
from eir.setup.output_setup_modules.survival_output_setup import (
    ComputedSurvivalOutputInfo,
)
from eir.setup.output_setup_modules.tabular_output_setup import (
    ComputedTabularOutputInfo,
)
from eir.setup.schemas import BasicInterpretationConfig

if TYPE_CHECKING:
    from eir.interpretation.interpretation import SampleAttribution
    from eir.predict import PredictExperiment
    from eir.predict_modules.predict_attributions import (
        LoadedTrainExperimentMixedWithPredict,
    )
    from eir.setup.input_setup_modules.setup_image import ImageNormalizationStats
    from eir.train import Experiment


def my_visualize_image_attr_multiple(*args, **kwargs):
    with plt.rc_context(
        {"figure.constrained_layout.use": False, "figure.autolayout": False}
    ):
        return visualize_image_attr_multiple(*args, **kwargs)


def analyze_image_input_attributions(
    experiment: Union[
        "Experiment",
        "PredictExperiment",
        "LoadedTrainExperimentMixedWithPredict",
    ],
    input_name: str,
    output_name: str,
    target_info: TargetTypeInfo,
    attribution_outfolder: Path,
    all_attributions: Sequence["SampleAttribution"],
) -> None:
    exp = experiment

    output_object = exp.outputs[output_name]
    assert isinstance(
        output_object, ComputedTabularOutputInfo | ComputedSurvivalOutputInfo
    )

    target_transformer = get_appropriate_target_transformer(
        output_object=output_object,
        target_column_name=target_info.name,
        target_column_type=target_info.type_,
    )

    input_object = exp.inputs[input_name]
    assert isinstance(input_object, ComputedImageInputInfo)

    interpretation_config = input_object.input_config.interpretation_config
    assert isinstance(interpretation_config, BasicInterpretationConfig)

    samples_to_act_analyze_gen = get_basic_sample_attributions_to_analyse_generator(
        interpretation_config=interpretation_config,
        all_attributions=all_attributions,
    )

    for sample_attribution in samples_to_act_analyze_gen:
        sample_target_labels = sample_attribution.sample_info.target_labels

        cur_label_name = get_target_class_name(
            sample_label=sample_target_labels[output_name][target_info.name],
            target_transformer=target_transformer,
            target_info=target_info,
        )

        attributions = sample_attribution.sample_attributions[input_name].squeeze(0)
        raw_input = sample_attribution.raw_inputs[input_name].cpu().numpy().squeeze(0)
        raw_input = un_normalize_image(
            normalized_img=raw_input,
            normalization_stats=input_object.normalization_stats,
        )
        raw_input = raw_input.transpose(1, 2, 0)
        attributions = attributions.transpose(1, 2, 0)

        figure, _ = my_visualize_image_attr_multiple(
            attr=attributions,
            original_image=raw_input,
            methods=["original_image", "heat_map"],
            signs=["all", "absolute_value"],
            show_colorbar=True,
            use_pyplot=False,
        )

        name = f"image_{sample_attribution.sample_info.ids[0]}_{cur_label_name}.pdf"
        output_path = attribution_outfolder / "single_samples" / name
        ensure_path_exists(path=output_path)
        figure.savefig(output_path, dpi=300)
        plt.close("all")


def un_normalize_image(
    normalized_img: np.ndarray, normalization_stats: "ImageNormalizationStats"
) -> np.ndarray:
    """
    Clip because sometimes we get values like 1.0000001 which will cause some libraries
    to do =/ 255, resulting in a black image.
    """
    means = torch.Tensor(normalization_stats.means)
    stds = torch.Tensor(normalization_stats.stds)
    un_norm = Normalize(
        mean=(-means / stds).tolist(),
        std=(1.0 / stds).tolist(),
    )

    img_as_tensor = torch.Tensor(normalized_img)
    img_un_normalized = un_norm(img_as_tensor).cpu().numpy()
    img_final = img_un_normalized.clip(0.0, 1.0)

    return img_final
