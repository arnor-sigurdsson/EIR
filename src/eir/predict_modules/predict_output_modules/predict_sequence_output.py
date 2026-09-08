from typing import TYPE_CHECKING

from eir.train_utils.evaluation_modules.train_handlers_sequence_output import (
    sequence_out_single_sample_evaluation_wrapper,
)

if TYPE_CHECKING:
    from eir.predict import PredictExperiment


def predict_sequence_wrapper(
    predict_experiment: "PredictExperiment",
    output_folder: str,
) -> None:
    """
    Note that this does in theory have access to all samples in the test dataset,
    but we do not run the sequence generation on the entire thing. Instead,
    what is actually run through the sequence generation is limited by
    the n_eval_inputs in the sampling config, so we do not end up saving
    thousands of samples to the disk accidentally.
    """
    sequence_out_single_sample_evaluation_wrapper(
        experiment=predict_experiment,
        iteration=0,
        input_objects=predict_experiment.inputs,
        auto_dataset_to_load_from=predict_experiment.test_dataset,
        output_folder=output_folder,
    )
