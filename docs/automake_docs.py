import argparse
import datetime
import os
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
from docs.doc_modules.i_scaling import (
    a_streaming_data,
    b_scaling_compute,
    c_scaling_diffusion,
)
from docs.doc_modules.serving_experiments import (
    AutoDocServingInfo,
    make_serving_tutorial_data,
)
from docs.doc_scripts.doc_performance_report import (
    collect_performance_data,
    generate_report,
)
from eir.utils.logging import get_logger

logger = get_logger(name=__name__)


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


def get_i_scaling_experiments() -> Iterable[AutoDocExperimentInfo]:
    a_experiments = a_streaming_data.get_experiments()
    b_experiments = b_scaling_compute.get_experiments()
    c_experiments = c_scaling_diffusion.get_experiments()

    return chain(
        a_experiments,
        b_experiments,
        c_experiments,
    )


def generate_performance_report(root_dir, output_dir):
    if not os.path.exists(root_dir):
        logger.error(f"Error: Root directory '{root_dir}' does not exist.")
        return

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    logger.info(f"Collecting performance data from {root_dir}...")

    current_date = datetime.datetime.now().strftime("%Y-%m-%d")
    output_file = os.path.join(output_dir, f"{current_date}.json")

    experiment_data = collect_performance_data(root_dir=root_dir)

    report = {
        "metadata": {
            "num_experiments": len(experiment_data),
            "date_generated": datetime.datetime.now().isoformat(),
        },
        "experiments": experiment_data,
    }

    report_content = generate_report(data=report, output_format="json")

    with open(output_file, "w") as f:
        f.write(report_content)

    logger.info(f"Performance report generated at {output_file}")
    logger.info(f"Found {len(experiment_data)} experiments with valid metrics.")


def parse_args():
    parser = argparse.ArgumentParser(description="Generate EIR documentation.")
    parser.add_argument(
        "--experiments",
        type=str,
        default=None,
        help="Comma-separated list of experiment groups to run (e.g., 'a,b,c'). "
        "Valid groups: a, b, c, d, e, f, g, h, i. "
        "If not specified, all experiments will be run.",
    )
    parser.add_argument(
        "--generate-report",
        action="store_true",
        help="Generate performance report after running experiments",
    )
    parser.add_argument(
        "--root-dir",
        default="eir_tutorials/tutorial_runs",
        help="Root directory containing the tutorial runs",
    )
    parser.add_argument(
        "--output-dir",
        default="docs/doc_data",
        help="Directory to save the performance report",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    if args.experiments is None:
        run_a = run_b = run_c = run_d = run_e = run_f = run_g = run_h = run_i = True
    else:
        experiment_ids = args.experiments.split(",")
        run_a = "a" in experiment_ids
        run_b = "b" in experiment_ids
        run_c = "c" in experiment_ids
        run_d = "d" in experiment_ids
        run_e = "e" in experiment_ids
        run_f = "f" in experiment_ids
        run_g = "g" in experiment_ids
        run_h = "h" in experiment_ids
        run_i = "i" in experiment_ids

    experiments = []

    if run_a:
        a_using_eir_experiments = _get_a_using_eir_experiments()
        experiments.append(a_using_eir_experiments)

    if run_c:
        c_sequence_outputs_experiments = _get_c_sequence_outputs_experiments()
        experiments.append(c_sequence_outputs_experiments)

    if run_b:
        b_customizing_eir_experiments = _get_b_customizing_eir_experiments()
        experiments.append(b_customizing_eir_experiments)

    if run_d:
        d_array_outputs_experiments = _get_d_array_outputs_experiments()
        experiments.append(d_array_outputs_experiments)

    if run_e:
        e_pretraining_experiments = _get_e_pretraining_outputs_experiments()
        experiments.append(e_pretraining_experiments)

    if run_f:
        f_image_outputs_experiments = _get_f_image_outputs_experiments()
        experiments.append(f_image_outputs_experiments)

    if run_g:
        g_time_series_experiments = get_g_time_series_experiments()
        experiments.append(g_time_series_experiments)

    if run_h:
        h_survival_analysis_experiments = get_h_survival_analysis_experiments()
        experiments.append(h_survival_analysis_experiments)

    if run_i:
        i_scaling_experiments = get_i_scaling_experiments()
        experiments.append(i_scaling_experiments)

    experiment_iter = chain.from_iterable(experiments)
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

    if args.generate_report:
        generate_performance_report(root_dir=args.root_dir, output_dir=args.output_dir)
