from itertools import chain

from docs.doc_modules import a_basic_tutorial, b_tabular_tutorial
from docs.doc_modules.experiments import make_tutorial_data

if __name__ == "__main__":
    a_experiments = a_basic_tutorial.get_experiments()
    b_experiments = b_tabular_tutorial.get_experiments()

    experiment_iter = chain.from_iterable(
        [
            a_experiments,
            b_experiments,
        ]
    )
    for experiment in experiment_iter:
        make_tutorial_data(auto_doc_experiment_info=experiment)
