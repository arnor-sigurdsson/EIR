from typing import TYPE_CHECKING

from eir.train_utils.evaluation_handlers.train_handlers_sequence_output import (
    sequence_out_single_sample_evaluation_wrapper,
)

if TYPE_CHECKING:
    from eir.predict import PredictExperiment


def predict_sequence_wrapper(
    predict_experiment: "PredictExperiment",
    output_folder: str,
) -> None:
    sequence_out_single_sample_evaluation_wrapper(
        experiment=predict_experiment,
        iteration=0,
        input_objects=predict_experiment.inputs,
        auto_dataset_to_load_from=predict_experiment.test_dataset,
        output_folder=output_folder,
    )
