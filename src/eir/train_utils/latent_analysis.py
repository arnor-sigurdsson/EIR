import json
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import torch
from aislib.misc_utils import ensure_path_exists
from matplotlib import pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from torch import nn

if TYPE_CHECKING:
    from eir.train_utils.evaluation import EvaluationResults


@dataclass
class LatentHookOutput:
    name: str
    all_sample_count: int
    output_folder: Path


def get_output_folder(
    run_folder: Path,
    iteration: int | str,
    layer_name: str,
) -> Path:
    folder = run_folder / "latents/latent_outputs" / f"{iteration}" / layer_name
    ensure_path_exists(path=folder, is_folder=True)
    return folder


def latent_analysis_wrapper(
    latent_outputs: LatentHookOutput,
    max_samples_for_viz: int | None,
) -> None:
    plot_pca_from_files(
        batch_dir=latent_outputs.output_folder,
        folder=latent_outputs.output_folder,
        max_samples=max_samples_for_viz,
    )

    plot_tsne_from_files(
        batch_dir=latent_outputs.output_folder,
        folder=latent_outputs.output_folder,
        max_samples=max_samples_for_viz,
    )


def register_latent_hook(
    model: nn.Module,
    layer_path: str,
    run_folder: Path,
    iteration: int | str,
    batch_size_for_saving: int = 1000,
) -> Callable[["EvaluationResults"], LatentHookOutput]:
    output_folder = get_output_folder(run_folder, iteration, layer_path)

    output_cache = []
    sample_indices = []
    batch_counter = 0
    total_samples = 0
    batch_to_indices = {}

    def hook(
        module: nn.Module,
        input: tuple[torch.Tensor, ...],
        output: torch.Tensor,
    ) -> None:
        nonlocal output_cache, sample_indices, batch_counter, total_samples

        batch_output = output.data.cpu().numpy()
        batch_size = batch_output.shape[0]

        output_cache.append(batch_output)

        current_indices = list(range(total_samples, total_samples + batch_size))
        sample_indices.extend(current_indices)

        total_samples += batch_size

        if sum(b.shape[0] for b in output_cache) >= batch_size_for_saving:
            combined_batch = np.concatenate(output_cache, axis=0)

            save_batch(
                output_dir=output_folder,
                batch_data=combined_batch,
                batch_num=batch_counter,
            )

            batch_to_indices[batch_counter] = sample_indices

            batch_counter += 1
            output_cache = []
            sample_indices = []

    layer = model
    for name in layer_path.split("."):
        layer = getattr(layer, name)

    if not isinstance(layer, nn.Module):
        raise ValueError(f"Layer {layer_path} not found.")

    handle = layer.register_forward_hook(hook=hook)

    def get_outputs_and_remove_hook(
        evaluation_results: "EvaluationResults",
    ) -> LatentHookOutput:
        nonlocal output_cache, sample_indices, batch_counter, total_samples
        nonlocal batch_to_indices

        handle.remove()

        if output_cache:
            combined_batch = np.concatenate(output_cache, axis=0)

            save_batch(
                output_dir=output_folder,
                batch_data=combined_batch,
                batch_num=batch_counter,
            )

            batch_to_indices[batch_counter] = sample_indices

            batch_counter += 1
            output_cache = []
            sample_indices = []

        all_ids = evaluation_results.all_ids

        for batch_num, indices in batch_to_indices.items():
            batch_file = output_folder / f"batch_{batch_num:05d}.npy"
            batch_data = np.load(batch_file)

            batch_ids = [all_ids[idx] for idx in indices]

            max_str_len = max(len(id_) for id_ in batch_ids)
            structured_array = np.empty(
                len(batch_data),
                dtype=[
                    ("Latent", float, batch_data.shape[1:]),
                    ("ID", f"U{max_str_len}"),
                ],
            )
            structured_array["Latent"] = batch_data
            structured_array["ID"] = batch_ids

            np.save(str(batch_file), structured_array)

        metadata = {
            "total_samples": total_samples,
            "batch_count": batch_counter,
            "batches": {
                batch_num: len(indices)
                for batch_num, indices in batch_to_indices.items()
            },
        }
        with open(output_folder / "metadata.json", "w") as f:
            json.dump(metadata, f, indent=2)

        latent_output_object = LatentHookOutput(
            name=layer_path,
            all_sample_count=total_samples,
            output_folder=output_folder,
        )
        return latent_output_object

    return get_outputs_and_remove_hook


def save_batch(output_dir: Path, batch_data: np.ndarray, batch_num: int) -> None:
    batch_file = output_dir / f"batch_{batch_num:05d}.npy"
    np.save(str(batch_file), batch_data)


def get_batch_files(batch_dir: Path) -> list[Path]:
    return sorted(batch_dir.glob("batch_*.npy"))


def load_samples_for_viz(
    batch_dir: Path,
    max_samples: int | None = None,
) -> tuple[np.ndarray, list[str]]:
    batch_files = sorted(batch_dir.glob("batch_*.npy"))
    batch_files = [f for f in batch_files if "_ids" not in f.name]

    if not batch_files:
        raise FileNotFoundError(f"No batch files found in {batch_dir}")

    collected_latents = []
    collected_ids = []
    samples_loaded = 0

    if max_samples is None:
        total_samples = 0
        for batch_file in batch_files:
            data = np.load(batch_file)
            total_samples += data.shape[0]
        effective_max = total_samples
    else:
        effective_max = max_samples

    for batch_file in batch_files:
        batch_data = np.load(batch_file)
        batch_size = batch_data.shape[0]

        if samples_loaded + batch_size <= effective_max:
            latents = batch_data["Latent"]
            ids = batch_data["ID"].tolist()

            collected_latents.append(latents)
            collected_ids.extend(ids)
            samples_loaded += batch_size
        else:
            samples_needed = effective_max - samples_loaded
            latents = batch_data["Latent"][:samples_needed]
            ids = batch_data["ID"][:samples_needed].tolist()

            collected_latents.append(latents)
            collected_ids.extend(ids)
            break

    return np.concatenate(collected_latents, axis=0), collected_ids


def parse_latent_outputs(latent_outputs: np.ndarray) -> np.ndarray:
    return latent_outputs.reshape(latent_outputs.shape[0], -1)


def plot_pca_from_files(batch_dir: Path, folder: Path, max_samples: int | None) -> None:
    latents, _ = load_samples_for_viz(
        batch_dir=batch_dir,
        max_samples=max_samples,
    )
    latents_parsed = parse_latent_outputs(latent_outputs=latents)

    pca = PCA(n_components=2, random_state=42)

    reduced = pca.fit_transform(X=latents_parsed)

    plt.figure(figsize=(8, 6))
    plt.scatter(reduced[:, 0], reduced[:, 1], s=2)

    variance_explained = pca.explained_variance_ratio_

    plt.xlabel(f"PCA 1 ({variance_explained[0]:.2f})")
    plt.ylabel(f"PCA 2 ({variance_explained[1]:.2f})")

    plt.title(f"PCA plot of latent space (using {latents.shape[0]} samples)")

    plt.savefig(folder / "pca.png")
    plt.close()


def plot_tsne_from_files(
    batch_dir: Path, folder: Path, max_samples: int | None
) -> None:
    latents, _ = load_samples_for_viz(
        batch_dir=batch_dir,
        max_samples=max_samples,
    )
    latents_parsed = parse_latent_outputs(latent_outputs=latents)

    tsne = TSNE(n_components=2, random_state=42)
    reduced = tsne.fit_transform(X=latents_parsed)

    plt.figure(figsize=(8, 6))
    plt.scatter(reduced[:, 0], reduced[:, 1], s=2)

    plt.xlabel("t-SNE 1")
    plt.ylabel("t-SNE 2")

    plt.title(f"t-SNE plot of latent space (using {latents.shape[0]} samples)")

    plt.savefig(folder / "tsne.png")
    plt.close()
