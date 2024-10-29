import base64
from typing import TYPE_CHECKING, Any

import numpy as np
import torch
from scipy.special import softmax
from sklearn.preprocessing import StandardScaler

from eir.models.model_setup_modules.input_model_setup.input_model_setup_sequence import (  # noqa
    get_sequence_model,
)
from eir.models.model_setup_modules.output_model_setup_modules.output_model_setup_array import (  # noqa
    get_array_or_image_output_module_from_model_config,
)
from eir.models.model_setup_modules.output_model_setup_modules.output_model_setup_sequence import (  # noqa
    get_sequence_output_module_from_model_config,
)
from eir.models.model_setup_modules.output_model_setup_modules.output_model_setup_tabular import (  # noqa
    get_tabular_output_module_from_model_config,
)
from eir.models.model_training_utils import recursive_to_device
from eir.models.output.sequence.sequence_output_modules import (
    SequenceOutputModuleConfig,
)
from eir.setup import schemas
from eir.setup.config_setup_modules.config_setup_utils import object_to_primitives
from eir.setup.input_setup import al_input_objects_as_dict
from eir.setup.input_setup_modules.setup_sequence import ComputedSequenceInputInfo
from eir.setup.output_setup import al_output_objects_as_dict
from eir.setup.output_setup_modules.array_output_setup import ComputedArrayOutputInfo
from eir.setup.output_setup_modules.image_output_setup import ComputedImageOutputInfo
from eir.setup.output_setup_modules.sequence_output_setup import (
    ComputedSequenceOutputInfo,
)
from eir.setup.output_setup_modules.survival_output_setup import (
    ComputedSurvivalOutputInfo,
)
from eir.setup.output_setup_modules.tabular_output_setup import (
    ComputedTabularOutputInfo,
)
from eir.train_utils.evaluation_modules.evaluation_handlers_utils import (
    remove_special_tokens,
)
from eir.train_utils.evaluation_modules.train_handlers_sequence_output import (
    decode_tokens,
    get_special_tokens,
    remove_special_tokens_from_string,
)
from eir.utils.logging import get_logger

if TYPE_CHECKING:
    pass

logger = get_logger(name=__name__)


def general_post_process(
    outputs: dict[str, dict[str, torch.Tensor | list[int] | np.ndarray]],
    output_objects: al_output_objects_as_dict,
    input_objects: al_input_objects_as_dict,
) -> dict[str, Any]:
    """
    Note that we always expect two levels for the predictions, in the case
    of tabular that's the output name and the column name, in the case of
    sequence we simply output name twice to conform to the same structure.

    Generally:
        - tabular outputs: torch.Tensor
        - sequence outputs: list[int]
        - array outputs: np.ndarray
        - image outputs: np.ndarray
        - survival outputs: torch.Tensor
    """
    post_processed: dict[str, Any] = {}

    outputs = recursive_to_device(obj=outputs, device="cpu")

    for output_name, output_object in output_objects.items():
        output_model_config = output_object.output_config.model_config
        cur_model_outputs = outputs[output_name]

        match output_object:
            case ComputedTabularOutputInfo():
                assert isinstance(
                    output_model_config, schemas.TabularOutputModuleConfig
                )

                tabular_outputs = _ensure_streamlined_tabular_values(
                    tabular_model_outputs=cur_model_outputs
                )

                tabular_inverse_transformed = _post_process_tabular_output(
                    output_object=output_object,
                    tabular_outputs=tabular_outputs,
                )
                post_processed[output_name] = tabular_inverse_transformed

            case ComputedSequenceOutputInfo():
                assert isinstance(output_model_config, SequenceOutputModuleConfig)

                output_type_info = output_object.output_config.output_type_info
                assert isinstance(output_type_info, schemas.SequenceOutputTypeConfig)

                generated_tokens = cur_model_outputs[output_name]
                cur_input_object = input_objects[output_name]
                assert isinstance(cur_input_object, ComputedSequenceInputInfo)
                assert cur_input_object.tokenizer is not None

                special_tokens = get_special_tokens(
                    tokenizer=cur_input_object.tokenizer,
                    vocab=cur_input_object.vocab,
                )

                assert isinstance(generated_tokens, list)
                generated_tokens = remove_special_tokens(
                    tokens=generated_tokens,
                    special_tokens=special_tokens,
                )

                generated_sample = decode_tokens(
                    tokens=generated_tokens,
                    vocab=cur_input_object.vocab,
                    split_on=output_type_info.split_on,
                )

                generated_sample = remove_special_tokens_from_string(
                    string=generated_sample,
                    special_tokens=special_tokens,
                )

                post_processed[output_name] = generated_sample

            case ComputedArrayOutputInfo():
                assert isinstance(output_model_config, schemas.ArrayOutputModuleConfig)

                array_np = cur_model_outputs[output_name]
                assert isinstance(array_np, np.ndarray)

                array_base64 = _post_process_array_outputs(array=array_np)
                post_processed[output_name] = array_base64

            case ComputedImageOutputInfo():
                assert isinstance(output_model_config, schemas.ArrayOutputModuleConfig)

                array_np = cur_model_outputs[output_name]
                assert isinstance(array_np, np.ndarray)

                array_base64 = _post_process_array_outputs(array=array_np)
                post_processed[output_name] = array_base64

            case ComputedSurvivalOutputInfo():
                assert isinstance(
                    output_model_config, schemas.TabularOutputModuleConfig
                )

                output_type_info = output_object.output_config.output_type_info
                assert isinstance(output_type_info, schemas.SurvivalOutputTypeConfig)

                event_name = output_type_info.event_column

                processed_outputs = process_survival_prediction(
                    output_object=output_object,
                    output_model_config=output_model_config,
                    cur_model_outputs=cur_model_outputs,
                )

                post_processed[output_name] = {}
                post_processed[output_name][event_name] = processed_outputs

            case _:
                raise NotImplementedError(
                    "Only tabular, sequence and array outputs are supported, got %s",
                    output_object,
                )

    post_processed = object_to_primitives(obj=post_processed)

    return post_processed


def _ensure_streamlined_tabular_values(
    tabular_model_outputs: dict[str, torch.Tensor | list[int] | np.ndarray]
) -> dict[str, torch.Tensor]:
    tensor_outputs = {
        k: v for k, v in tabular_model_outputs.items() if isinstance(v, torch.Tensor)
    }

    assert len(tensor_outputs) == len(tabular_model_outputs)

    return tensor_outputs


def _post_process_tabular_output(
    output_object: ComputedTabularOutputInfo, tabular_outputs: dict[str, torch.Tensor]
) -> dict[str, dict[str, float]]:
    processed_outputs: dict[str, dict[str, float]] = {}
    target_columns = output_object.target_columns

    for target_column_type, list_of_cols_of_this_type in target_columns.items():
        for cur_column in list_of_cols_of_this_type:
            if target_column_type == "con":
                cur_classes = [cur_column]
                cur_transformer = output_object.target_transformers[cur_column]
                cur_output_normalized = _normalize_continuous_outputs(
                    outputs=tabular_outputs[cur_column], transformer=cur_transformer
                )
                cur_output_normalized = cur_output_normalized
            elif target_column_type == "cat":
                cur_classes = output_object.target_transformers[cur_column].classes_
                cur_output_normalized = _normalize_categorical_outputs(
                    outputs=tabular_outputs[cur_column]
                )
            else:
                raise ValueError(
                    "Expected column type to be con or cat, but got %s",
                    target_column_type,
                )

            cur_class_to_confidence_mapping = {
                k: v for k, v in zip(cur_classes, cur_output_normalized)
            }

            processed_outputs[cur_column] = cur_class_to_confidence_mapping

    return processed_outputs


def _normalize_categorical_outputs(outputs: torch.Tensor) -> tuple[float]:
    cur_outputs_normalized = softmax(outputs).squeeze().tolist()

    return tuple(cur_outputs_normalized)


def _normalize_continuous_outputs(
    outputs: torch.Tensor, transformer: StandardScaler
) -> tuple[float]:
    cur_output_reshaped = outputs.reshape(1, -1)
    transform_func = transformer.inverse_transform
    cur_output_normalized = transform_func(cur_output_reshaped).squeeze()
    cur_outputs_normalized = (cur_output_normalized.item(),)

    return cur_outputs_normalized


def _post_process_array_outputs(array: np.ndarray) -> str:
    base64_encoded = _serialize_array_to_base64(array=array)
    return base64_encoded


def _serialize_array_to_base64(array: np.ndarray) -> str:
    array_bytes = array.tobytes()
    base64_encoded = base64.b64encode(array_bytes).decode("utf-8")
    return base64_encoded


def process_survival_prediction(
    output_object: ComputedSurvivalOutputInfo,
    output_model_config: schemas.TabularOutputModuleConfig,
    cur_model_outputs: dict[str, torch.Tensor | list[int] | np.ndarray],
) -> dict:

    assert isinstance(output_model_config, schemas.TabularOutputModuleConfig)
    assert isinstance(
        output_object.output_config.output_type_info, schemas.SurvivalOutputTypeConfig
    )

    tabular_outputs = _ensure_streamlined_tabular_values(
        tabular_model_outputs=cur_model_outputs
    )

    output_type_info = output_object.output_config.output_type_info
    model_type = "cox" if output_type_info.loss_function == "CoxPHLoss" else "discrete"

    if model_type == "discrete":
        processed_outputs = _process_discrete_survival_prediction(
            output_object=output_object,
            cur_model_outputs=tabular_outputs,
        )
    else:
        processed_outputs = _process_cox_survival_prediction(
            output_object=output_object,
            cur_model_outputs=tabular_outputs,
        )

    return processed_outputs


def _process_discrete_survival_prediction(
    output_object: ComputedSurvivalOutputInfo,
    cur_model_outputs: dict[str, torch.Tensor],
) -> dict:

    processed_outputs: dict[str, dict[str, Any]] = {}

    output_type_info = output_object.output_config.output_type_info
    assert isinstance(output_type_info, schemas.SurvivalOutputTypeConfig)

    time_name = output_type_info.time_column
    event_name = output_type_info.event_column

    transformers = output_object.target_transformers
    time_kbins_transformer = transformers[time_name]
    time_bins = time_kbins_transformer.bin_edges_[0]
    time_bins_except_last = time_bins[:-1]

    hazards_logits = cur_model_outputs[event_name]
    hazards = torch.sigmoid(hazards_logits).numpy()
    survival_probs = np.cumprod(1 - hazards, 0)

    processed_outputs["time_points"] = time_bins_except_last.tolist()
    processed_outputs["survival_probs"] = survival_probs.tolist()

    return processed_outputs


def _process_cox_survival_prediction(
    output_object: ComputedSurvivalOutputInfo,
    cur_model_outputs: dict[str, torch.Tensor],
) -> dict:

    processed_outputs: dict[str, dict[str, Any]] = {}

    output_type_info = output_object.output_config.output_type_info
    assert isinstance(output_type_info, schemas.SurvivalOutputTypeConfig)

    event_name = output_type_info.event_column

    risk_scores = cur_model_outputs[event_name].numpy()

    msg_1 = "Baseline hazard not found. Make sure it was computed during validation."
    assert output_object.baseline_hazard is not None, msg_1

    msg_2 = (
        "Baseline unique times not found. Make sure they were computed "
        "during validation."
    )
    assert output_object.baseline_unique_times is not None, msg_2

    baseline_survival = np.exp(-np.cumsum(output_object.baseline_hazard))

    max_time = output_object.baseline_unique_times[-1]
    time_points = np.linspace(0, max_time, 100)

    interpolated_baseline = np.interp(
        time_points,
        output_object.baseline_unique_times,
        baseline_survival,
        right=baseline_survival[-1],
    )

    survival_probs = np.zeros((len(risk_scores), len(time_points)))
    for i, risk_score in enumerate(risk_scores):
        survival_probs[i] = interpolated_baseline ** np.exp(risk_score)

    processed_outputs["survival_probs"] = survival_probs.tolist()
    processed_outputs["time_points"] = time_points.tolist()

    return processed_outputs
