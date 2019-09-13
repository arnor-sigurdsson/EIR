from functools import partial
from functools import wraps
from pathlib import Path
from typing import Dict, Tuple, Union, List, TYPE_CHECKING

import numpy as np
import shap
from aislib.misc_utils import ensure_path_exists, get_logger
from ignite.engine import Engine
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import matthews_corrcoef
from torch.utils.data import DataLoader

from human_origins_supervised.models import model_utils
from human_origins_supervised.visualization import model_visualization as mv
from human_origins_supervised.visualization import visualization_funcs as vf

if TYPE_CHECKING:
    from human_origins_supervised.train import Config

logger = get_logger(__name__)


def func_call_logger(pre_message, post_message):
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            logger.info(pre_message)
            result = func(*args, **kwargs)
            logger.info(post_message)

            return result

        return wrapper

    return decorator


def flatten(array: np.ndarray) -> np.ndarray:
    array = array.squeeze()
    if len(array.shape) < 3:
        array = np.expand_dims(array, 0)

    flattened_arr = np.reshape(array, (array.shape[0], -1), order="F")

    return flattened_arr


def predict_on_array(
    classifier: Union[LogisticRegression, SGDClassifier], array: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    probs = classifier.predict_proba(array)
    preds = probs.argmax(axis=1)

    return probs, preds


def predict_sklearn_on_supervised_loader(
    classifier: SGDClassifier, loader: DataLoader
) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """
    Sometimes we get large negatives for all classes after the linear combination,
    so when we pass those to the logistic function (expit(x) = 1/(1+exp(-x))) we
    get 0s for all classes. In the OvR normalization of sci-kit learn we do
    prob /= prob.sum(axis=1).reshape((prob.shape[0], -1)) so we end up with a bunch
    of NaNs if prob is an array of 0s. Hence we threshold them to 0 here.
    """
    outputs_total = []
    labels_total = []
    ids_total = []
    for inputs, labels, ids in loader:
        inputs = flatten(inputs.cpu().numpy())
        labels = labels.cpu().numpy()

        outputs = classifier.predict_proba(inputs)
        outputs[np.isnan(outputs)] = 0

        outputs_total += [i for i in outputs]
        labels_total += [i for i in labels]
        ids_total += [i for i in ids]

    return np.array(outputs_total), np.array(labels_total), ids_total


def save_bencmark_metrics(outfolder: Path, metrics: Dict[str, float]) -> None:
    with open(str(outfolder / "benchmark_metrics.txt"), "w") as outfile:
        for metric, value in metrics.items():
            outfile.write(f"{metric}: {value}" + "\n")


def gather_and_flatten_samples(
    dloader: DataLoader, n_samples=None
) -> Tuple[np.ndarray, np.ndarray]:
    gatherer = model_utils.gather_dloader_samples

    tensors, labels, *_ = gatherer(dloader, "cpu", n_samples=n_samples)

    array = tensors.numpy().squeeze()

    return flatten(array), np.array(labels)


def run_partial_fit(
    classifier: SGDClassifier,
    training_dloader: DataLoader,
    classes: np.ndarray,
    n_epochs: int,
) -> SGDClassifier:

    counter = 1
    percentiles = [int(i) for i in np.percentile(range(n_epochs), [25, 50, 75, 100])]
    for n in range(n_epochs):

        if n in percentiles:
            percent = int(0.25 * 100 * counter)
            logger.info("Logistic regression fit %d%% done.", percent)
            counter += 1

        for inputs, labels, *_ in training_dloader:
            inputs = flatten(inputs.cpu().numpy())
            labels = labels.cpu().numpy()

            classifier.partial_fit(inputs, labels, classes)
    return classifier


@func_call_logger(
    "Running benchmark with logistic regression.",
    "Benchmark with logistic regression finished.",
)
def benchmark(engine: Engine, config: "Config", run_folder: Path) -> None:
    c = config

    benchmark_folder = Path(run_folder, "benchmark")
    ensure_path_exists(benchmark_folder, is_folder=True)

    lr_classifier = SGDClassifier(loss="log", n_jobs=-1)

    def flatten_hook(single_sample, sample_label) -> tuple:
        return flatten(single_sample), sample_label

    def linear_act_func(single_sample, sample_label, explainer_):
        """
        Note that in binary cases, activations for sample being the positive
        class by default. Hence we multiply by -1 to invert the activations
        for the negative class.
        """

        acts = explainer_.shap_values(single_sample)
        if len(acts) == 1:
            if sample_label.item() == 0:
                return acts * -1
            return acts
        return acts[sample_label]

    def post_reshape(flat_activations):
        reshaped_activations = np.reshape(
            flat_activations, (-1, c.data_width), order="F"
        )

        return reshaped_activations

    lr_classifier = run_partial_fit(
        lr_classifier,
        c.train_loader,
        np.arange(len(c.label_encoder.classes_)),
        c.cl_args.n_epochs,
    )

    predictor = predict_sklearn_on_supervised_loader
    lr_val_probs, valid_labels, valid_ids = predictor(lr_classifier, c.valid_loader)
    lr_val_preds = np.argmax(lr_val_probs, axis=1)

    lr_train_probs, train_labels, train_ids = predictor(lr_classifier, c.train_loader)
    lr_train_preds = np.argmax(lr_train_probs, axis=1)

    lr_val_mcc = matthews_corrcoef(valid_labels, lr_val_preds)
    lr_train_mcc = matthews_corrcoef(train_labels, lr_train_preds)

    # save analysis
    metric_results = {"VAL MCC": lr_val_mcc, "TRAIN MCC": lr_train_mcc}

    save_bencmark_metrics(benchmark_folder, metric_results)

    vf.gen_eval_graphs(
        valid_labels,
        lr_val_probs,
        valid_ids,
        benchmark_folder,
        c.label_encoder,
        config.cl_args.model_task,
    )

    # analyze activations
    if c.cl_args.get_acts:
        train_background, _ = gather_and_flatten_samples(c.train_loader, 1000)
        explainer = shap.LinearExplainer(
            lr_classifier, train_background, feature_dependence="independent"
        )

        proc_funcs = {"pre": (flatten_hook,), "post": (post_reshape,)}

        act_func = partial(linear_act_func, explainer_=explainer)

        mv.analyze_activations(config, act_func, proc_funcs, benchmark_folder)
