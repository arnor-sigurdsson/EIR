from docs.doc_modules import a_basic_tutorial
from docs.doc_modules.experiments import make_tutorial_data

if __name__ == "__main__":
    a_experiments = a_basic_tutorial.get_experiments()

    for experiment in a_experiments:
        make_tutorial_data(auto_doc_experiment_info=experiment)
