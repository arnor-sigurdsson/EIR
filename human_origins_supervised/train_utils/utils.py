from typing import List, Dict

from human_origins_supervised.data_load.label_setup import al_label_dict


def get_extra_labels_from_ids(
    labels_dict: al_label_dict, cur_ids: List[str], label_column: str
) -> List[Dict[str, str]]:
    """
    Returns a batch in same order as cur_ids.
    """
    extra_labels = []
    for sample_id in cur_ids:
        cur_labels_all = labels_dict.get(sample_id)
        cur_labels_extra = {
            k: v for k, v in cur_labels_all.items() if k != label_column
        }
        extra_labels.append(cur_labels_extra)

    return extra_labels


def check_if_iteration_sample(
    iteration: int, iter_sample_interval: int, n_iterations_per_epochs, n_epochs
):
    if iter_sample_interval:
        condition_1 = iteration % iter_sample_interval == 0
    else:
        condition_1 = False

    condition_2 = iteration == n_iterations_per_epochs * n_epochs

    return condition_1 or condition_2
