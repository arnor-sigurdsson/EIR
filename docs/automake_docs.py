from collections.abc import Iterable
from itertools import chain

from docs.doc_modules.a_using_eir import (
    a_basic_tutorial,
    b_tabular_tutorial,
    c_sequence_tutorial,
    d_pretrained_models_tutorial,
    e_image_tutorial,
    f_binary_tutorial,
    g_multimodal_tutorial,
    h_array_tutorial,
)
from docs.doc_modules.api import generate_hf_sequence_info, generate_timm_api_info
from docs.doc_modules.b_customizing_eir import a_customizing_fusion_tutorial
from docs.doc_modules.c_sequence_outputs import (
    a_sequence_generation,
    b_sequence_to_sequence,
    c_image_captioning,
    d_protein_sequence_generation,
)
from docs.doc_modules.d_array_outputs import a_array_mnist_generation
from docs.doc_modules.e_pretraining import a_checkpointing, b_mini_foundation
from docs.doc_modules.experiments import (
    AutoDocExperimentInfo,
    make_training_or_predict_tutorial_data,
)
from docs.doc_modules.f_image_outputs import (
    a_image_foundation,
    b_image_colorization,
    c_mnist_diffusion,
)
from docs.doc_modules.g_time_series import a_time_series_power, b_time_series_stocks
from docs.doc_modules.h_survival_analysis import a_flchain, b_flchain_cox
from docs.doc_modules.serving_experiments import (
    AutoDocServingInfo,
    make_serving_tutorial_data,
)


def _get_a_using_eir_experiments() -> Iterable[AutoDocExperimentInfo]:
    a_experiments = a_basic_tutorial.get_experiments()
    b_experiments = b_tabular_tutorial.get_experiments()
    c_experiments = c_sequence_tutorial.get_experiments()
    d_experiments = d_pretrained_models_tutorial.get_experiments()
    e_experiments = e_image_tutorial.get_experiments()
    f_experiments = f_binary_tutorial.get_experiments()
    g_experiments = g_multimodal_tutorial.get_experiments()
    h_experiments = h_array_tutorial.get_experiments()

    return chain(
        a_experiments,
        b_experiments,
        c_experiments,
        d_experiments,
        e_experiments,
        f_experiments,
        g_experiments,
        h_experiments,
    )


def _get_b_customizing_eir_experiments() -> Iterable[AutoDocExperimentInfo]:
    a_experiments = a_customizing_fusion_tutorial.get_experiments()

    return chain(
        a_experiments,
    )


def _get_c_sequence_outputs_experiments() -> Iterable[AutoDocExperimentInfo]:
    a_experiments = a_sequence_generation.get_experiments()
    b_experiments = b_sequence_to_sequence.get_experiments()
    c_experiments = c_image_captioning.get_experiments()
    d_experiments = d_protein_sequence_generation.get_experiments()

    return chain(
        a_experiments,
        b_experiments,
        c_experiments,
        d_experiments,
    )


def _get_d_array_outputs_experiments() -> Iterable[AutoDocExperimentInfo]:
    a_experiments = a_array_mnist_generation.get_experiments()

    return chain(
        a_experiments,
    )


def _get_e_pretraining_outputs_experiments() -> Iterable[AutoDocExperimentInfo]:
    a_experiments = a_checkpointing.get_experiments()
    b_experiments = b_mini_foundation.get_experiments()

    return chain(
        a_experiments,
        b_experiments,
    )


def _get_f_image_outputs_experiments() -> Iterable[AutoDocExperimentInfo]:
    a_experiments = a_image_foundation.get_experiments()
    b_experiments = b_image_colorization.get_experiments()
    c_experiments = c_mnist_diffusion.get_experiments()

    return chain(
        a_experiments,
        b_experiments,
        c_experiments,
    )


def get_g_time_series_experiments() -> Iterable[AutoDocExperimentInfo]:
    a_experiments = a_time_series_power.get_experiments()
    b_experiments = b_time_series_stocks.get_experiments()

    return chain(
        a_experiments,
        b_experiments,
    )


def get_h_survival_analysis_experiments() -> Iterable[AutoDocExperimentInfo]:
    a_experiments = a_flchain.get_experiments()
    b_experiments = b_flchain_cox.get_experiments()

    return chain(
        a_experiments,
        b_experiments,
    )


if __name__ == "__main__":
    a_using_eir_experiments = _get_a_using_eir_experiments()
    c_sequence_outputs_experiments = _get_c_sequence_outputs_experiments()
    b_customizing_eir_experiments = _get_b_customizing_eir_experiments()
    d_array_outputs_experiments = _get_d_array_outputs_experiments()
    e_pretraining_experiments = _get_e_pretraining_outputs_experiments()
    f_image_outputs_experiments = _get_f_image_outputs_experiments()
    g_time_series_experiments = get_g_time_series_experiments()
    h_survival_analysis_experiments = get_h_survival_analysis_experiments()

    experiment_iter = chain.from_iterable(
        [
            a_using_eir_experiments,
            c_sequence_outputs_experiments,
            b_customizing_eir_experiments,
            d_array_outputs_experiments,
            e_pretraining_experiments,
            f_image_outputs_experiments,
            g_time_series_experiments,
            h_survival_analysis_experiments,
        ]
    )
    for experiment in experiment_iter:
        match experiment:
            case AutoDocExperimentInfo():
                make_training_or_predict_tutorial_data(
                    auto_doc_experiment_info=experiment
                )

            case AutoDocServingInfo():
                make_serving_tutorial_data(auto_doc_experiment_info=experiment)

    generate_timm_api_info.run_all()
    generate_hf_sequence_info.run_all()
