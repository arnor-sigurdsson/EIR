from itertools import chain

from docs.doc_modules import (
    a_basic_tutorial,
    b_tabular_tutorial,
    c_sequence_tutorial,
    d_pretrained_models_tutorial,
)
from docs.doc_modules.experiments import make_tutorial_data

if __name__ == "__main__":
    a_experiments = a_basic_tutorial.get_experiments()
    b_experiments = b_tabular_tutorial.get_experiments()
    c_experiments = c_sequence_tutorial.get_experiments()
    d_experiments = d_pretrained_models_tutorial.get_experiments()

    experiment_iter = chain.from_iterable(
        [
            # a_experiments,
            # b_experiments,
            # c_experiments,
            d_experiments
        ]
    )
    for experiment in experiment_iter:
        make_tutorial_data(auto_doc_experiment_info=experiment)
