from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Callable

import numpy as np
import torch
from aislib.misc_utils import ensure_path_exists
from matplotlib import pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from torch import nn
from umap import UMAP

if TYPE_CHECKING:
    from eir.train_utils.evaluation import EvaluationResults


@dataclass
class LatentHookOutput:
    name: str
    outputs: np.ndarray
    ids: list[str]


def latent_analysis_wrapper(
    latent_outputs: LatentHookOutput,
    run_folder: Path,
    iteration: int | str,
) -> None:
    latent_output_folder = (
        run_folder / "latents/latent_outputs" / f"{iteration}" / latent_outputs.name
    )
    ensure_path_exists(path=latent_output_folder, is_folder=True)

    save_latent_outputs(
        outputs=latent_outputs.outputs,
        ids=latent_outputs.ids,
        folder=latent_output_folder,
    )

    outputs_parsed = parse_latent_outputs(latent_outputs=latent_outputs.outputs)
    plot_pca(outputs=outputs_parsed, folder=latent_output_folder)
    plot_tsne(outputs=outputs_parsed, folder=latent_output_folder)
    plot_umap(outputs=outputs_parsed, folder=latent_output_folder)


def register_latent_hook(
    model: nn.Module,
    layer_path: str,
) -> Callable[["EvaluationResults"], LatentHookOutput]:
    outputs = []

    def hook(
        module: nn.Module,
        input: tuple[torch.Tensor, ...],
        output: torch.Tensor,
    ) -> None:
        outputs.append(output.data.cpu().numpy())

    layer = model
    for name in layer_path.split("."):
        layer = getattr(layer, name)

    if not isinstance(layer, nn.Module):
        raise ValueError(f"Layer {layer_path} not found.")

    handle = layer.register_forward_hook(hook=hook)

    def get_outputs_and_remove_hook(
        evaluation_results: "EvaluationResults",
    ) -> LatentHookOutput:
        handle.remove()
        outputs_arr = np.concatenate(outputs, axis=0)
        ids = evaluation_results.all_ids
        latent_output_object = LatentHookOutput(
            outputs=outputs_arr,
            ids=ids,
            name=layer_path,
        )
        return latent_output_object

    return get_outputs_and_remove_hook


def parse_latent_outputs(latent_outputs: np.ndarray) -> np.ndarray:
    return latent_outputs.reshape(latent_outputs.shape[0], -1)


def save_latent_outputs(
    outputs: np.ndarray, ids: list[str], folder: Path
) -> np.ndarray:
    assert len(outputs) == len(ids), "Number of outputs and IDs must match."

    ids_array = np.array(ids)
    max_str_len = max(len(id_) for id_ in ids_array)

    structured_array = np.empty(
        len(outputs),
        dtype=[
            ("Latent", float, outputs.shape[1:]),
            ("ID", f"U{max_str_len}"),
        ],
    )
    structured_array["Latent"] = outputs
    structured_array["ID"] = ids_array

    np.save(file=str(folder / "latents.npy"), arr=structured_array)
    return structured_array


def plot_pca(outputs: np.ndarray, folder: Path) -> None:
    pca = PCA(n_components=2)
    reduced = pca.fit_transform(X=outputs)

    plt.figure(figsize=(8, 6))
    plt.scatter(reduced[:, 0], reduced[:, 1], s=2)

    variance_explained = pca.explained_variance_ratio_

    plt.xlabel(f"PCA 1 ({variance_explained[0]:.2f})")
    plt.ylabel(f"PCA 2 ({variance_explained[1]:.2f})")

    plt.title("PCA plot of latent space")

    plt.savefig(folder / "pca.png")


def plot_tsne(outputs: np.ndarray, folder: Path) -> None:
    tsne = TSNE(n_components=2)
    reduced = tsne.fit_transform(X=outputs)

    plt.figure(figsize=(8, 6))
    plt.scatter(reduced[:, 0], reduced[:, 1], s=2)

    plt.xlabel("t-SNE 1")
    plt.ylabel("t-SNE 2")

    plt.title("t-SNE plot of latent space")

    plt.savefig(folder / "tsne.png")


def plot_umap(outputs: np.ndarray, folder: Path) -> None:
    umap = UMAP(n_components=2)

    reduced = umap.fit_transform(X=outputs)

    plt.figure(figsize=(8, 6))
    plt.scatter(reduced[:, 0], reduced[:, 1], s=2)

    plt.xlabel("UMAP 1")
    plt.ylabel("UMAP 2")
    plt.title("UMAP plot of latent space")

    plt.savefig(folder / "umap.png")
