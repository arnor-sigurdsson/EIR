from functools import partial
from functools import wraps
from pathlib import Path
from typing import Dict, Tuple, Union, List, TYPE_CHECKING

import numpy as np
import shap
from aislib.misc_utils import ensure_path_exists, get_logger
from ignite.engine import Engine
from sklearn.linear_model import SGDClassifier, SGDRegressor
from sklearn.metrics import matthews_corrcoef, r2_score
from torch.utils.data import DataLoader

from human_origins_supervised.models import model_utils
from human_origins_supervised.train_utils import activation_analysis as av
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
    classifier: Union[SGDRegressor, SGDClassifier], array: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    probs = classifier.predict_proba(array)
    preds = probs.argmax(axis=1)

    return probs, preds


def predict_sklearn_on_supervised_loader(
    classifier: Union[SGDClassifier, SGDRegressor], loader: DataLoader
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

        if isinstance(classifier, SGDClassifier):
            outputs = classifier.predict_proba(inputs)
        else:
            outputs = classifier.predict(inputs)

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
    classifier: Union[SGDClassifier, SGDRegressor],
    training_dloader: DataLoader,
    classes: np.ndarray,
    n_epochs: int,
) -> SGDClassifier:

    counter = 1
    percentiles = [int(i) for i in np.percentile(range(n_epochs), [25, 50, 75, 100])]
    for n in range(n_epochs):

        if n in percentiles:
            percent = int(0.25 * 100 * counter)
            logger.info("Linear fit %d%% done.", percent)
            counter += 1

        for inputs, labels, *_ in training_dloader:
            inputs = flatten(inputs.cpu().numpy())
            labels = labels.cpu().numpy()

            if isinstance(classifier, SGDClassifier):
                classifier.partial_fit(inputs, labels, classes)
            else:
                classifier.partial_fit(inputs, labels)

    return classifier


def get_linear_model_metrics(
    model_task, val_preds, val_labels, train_preds, train_labels
):

    if model_task == "cls":
        val_preds_max = np.argmax(val_preds, axis=1)
        train_preds_max = np.argmax(train_preds, axis=1)

        lr_val_mcc = matthews_corrcoef(val_labels, val_preds_max)
        lr_train_mcc = matthews_corrcoef(train_labels, train_preds_max)

        metric_results = {"v_mcc": lr_val_mcc, "t_mcc": lr_train_mcc}
    else:
        lr_val_r2 = r2_score(val_labels, val_preds)
        lr_train_r2 = r2_score(train_labels, train_preds)

        metric_results = {"v_r2": lr_val_r2, "t_r2": lr_train_r2}

    return metric_results


@func_call_logger(
    "Running benchmark with linear model.", "Benchmark with linear model finished."
)
def benchmark(engine: Engine, config: "Config", run_folder: Path) -> None:
    c = config

    benchmark_folder = Path(run_folder, "benchmark")
    ensure_path_exists(benchmark_folder, is_folder=True)

    linear_model = (
        SGDClassifier(loss="log", n_jobs=-1)
        if c.cl_args.model_task == "cls"
        else SGDRegressor()
    )

    def flatten_hook(single_sample, sample_label) -> tuple:
        return flatten(single_sample), sample_label

    def linear_act_func(inputs: av.al_model_inputs, sample_label, explainer_):
        """
        Note that in binary cases, activations for sample being the positive
        class by default. Hence we multiply by -1 to invert the activations
        for the negative class.

        Note that inputs here is composed of the SNP data and embeddings, but the
        linear benchmarking currently only supports the SNP data.
        """

        acts = explainer_.shap_values(inputs[0])
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

    classes = (
        np.arange(len(c.target_transformer.classes_))
        if c.cl_args.model_task == "cls"
        else None
    )
    linear_model = run_partial_fit(
        linear_model, c.train_loader, classes, c.cl_args.n_epochs
    )

    predictor = predict_sklearn_on_supervised_loader
    lin_val_preds, val_labels, val_ids = predictor(linear_model, c.valid_loader)
    lin_trn_preds, trn_labels, trn_ids = predictor(linear_model, c.train_loader)

    metric_results = get_linear_model_metrics(
        c.cl_args.model_task, lin_val_preds, val_labels, lin_trn_preds, trn_labels
    )

    save_bencmark_metrics(benchmark_folder, metric_results)

    vf.gen_eval_graphs(
        val_labels,
        lin_val_preds,
        val_ids,
        benchmark_folder,
        c.target_transformer,
        config.cl_args.model_task,
    )

    # analyze activations
    if c.cl_args.get_acts:
        train_background, _ = gather_and_flatten_samples(c.train_loader, 1000)
        explainer = shap.LinearExplainer(
            linear_model, train_background, feature_dependence="independent"
        )

        proc_funcs = {"pre": (flatten_hook,), "post": (post_reshape,)}

        act_func = partial(linear_act_func, explainer_=explainer)

        av.analyze_activations(config, act_func, proc_funcs, benchmark_folder)
