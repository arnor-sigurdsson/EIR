from typing import Iterable, Sequence

import torch
from tqdm import tqdm
import timm
from torch_optimizer import _NAME_OPTIM_MAP
from transformers.models.auto.modeling_auto import MODEL_MAPPING_NAMES


class RunningStatistics:
    """
    Adapted from https://gist.github.com/thomasbrandon/ad5b1218fc573c10ea4e1f0c63658469.
    """

    def __init__(self, n_dims: int = 2):
        self.n_dims = n_dims
        self.n = 0
        self.sum = 0

        self.shape = None
        self.num_var = None

    def update(self, data: torch.Tensor):
        data = data.view(*list(data.shape[: -self.n_dims]) + [-1])

        with torch.no_grad():
            new_n = data.shape[-1]
            new_var = data.var(-1)
            new_sum = data.sum(-1)

            if self.n == 0:
                self.n = new_n
                self.shape = data.shape[:-1]
                self.sum = new_sum
                self.num_var = new_var.mul_(new_n)

            else:

                ratio = self.n / new_n
                t = (self.sum / ratio).sub_(new_sum).pow_(2)

                self.num_var.add_(other=new_var, alpha=new_n)
                self.num_var.add_(other=t, alpha=ratio / (self.n + new_n))

                self.sum += new_sum
                self.n += new_n

    @property
    def mean(self):
        return self.sum / self.n if self.n > 0 else None

    @property
    def var(self):
        return self.num_var / self.n if self.n > 0 else None

    @property
    def std(self):
        return self.var.sqrt() if self.n > 0 else None


def collect_stats(
    tensor_iterable: Iterable[torch.Tensor], n_dims: int = 2
) -> RunningStatistics:
    stats = RunningStatistics(n_dims)
    for it in tqdm(tensor_iterable, desc="Gathering Image Statistics: "):
        if hasattr(it, "data"):
            stats.update(it.data)
        else:
            stats.update(it)
    return stats


def get_base_optimizer_names() -> set:
    base_names = {"sgdm", "adam", "adamw", "adahessian", "adabelief", "adabeliefw"}

    return base_names


def get_all_optimizer_names() -> Sequence[str]:
    external_optimizers = set(_NAME_OPTIM_MAP.keys())
    base_optimizers = get_base_optimizer_names()
    all_optimizers = set.union(base_optimizers, external_optimizers)
    all_optimizers = sorted(list(all_optimizers))

    return all_optimizers


def get_all_timm_model_names() -> Sequence[str]:
    pretrained_names = {i for i in timm.list_models() if not i.startswith("tf")}
    other_model_classes = {i for i in dir(timm.models) if "Net" in i}
    all_models = set.union(pretrained_names, other_model_classes)
    all_models_list = sorted(list(all_models))

    return all_models_list


def get_all_hf_model_names() -> Sequence[str]:
    all_models = sorted(list(MODEL_MAPPING_NAMES.keys()))
    unsupported = get_unsupported_hf_models()
    unsupported_names = unsupported.keys()
    return [i for i in all_models if i not in unsupported_names]


def get_unsupported_hf_models() -> dict:
    unsupported = {
        "audio-spectrogram-transformer": "Not strictly sequence model.",
        "beit": "Not strictly sequence model.",
        "canine": "Cannot do straightforward look up of embeddings.",
        "clip": "Not strictly sequence model.",
        "clipseg": "Not strictly sequence model.",
        "chinese_clip": "Not strictly sequence model.",
        "conditional_detr": "Not strictly sequence model.",
        "convbert": "HF error.",
        "convnext": "Not strictly sequence model..",
        "cvt": "Not strictly sequence model.",
        "data2vec-audio": "Not strictly sequence model.",
        "data2vec-vision": "Not strictly sequence model.",
        "decision_transformer": "Raises NotImplementedError",
        "decision_transformer_gpt2": "Raises ValueError: Unrecognized model identifier",
        "deformable_detr": "Not strictly sequence model.",
        "deit": "Not strictly sequence model.",
        "dinat": "Not strictly sequence model.",
        "detr": "Not strictly sequence model.",
        "donut-swin": "Not strictly sequence model.",
        "dpr": "Not strictly sequence model.",
        "dpt": "Not strictly sequence model.",
        "esm": "Needs fixing/special handling of init parameters.",
        "fsmt": "Not strictly sequence model.",
        "funnel": "HF error.",
        "flava": "Not strictly sequence model.",
        "nat": "Requires the natten library",
        "glpn": "Not strictly sequence model.",
        "groupvit": "Not strictly sequence model.",
        "gpt_neo": "Configuration troublesome w.r.t. attn layers matching num_layers.",
        "hubert": "Cannot do straightforward look up of embeddings.",
        "jukebox": "Not strictly sequence model.",
        "layoutlmv2": "Not strictly sequence model.",
        "layoutlmv3": "Not strictly sequence model.",
        "levit": "Not strictly sequence model.",
        "lilt": "Matmul incompatible shapes with default params.",
        "lxmert": "Not strictly sequence model.",
        "maskformer": "Not strictly sequence model.",
        "maskformer-swin": "Not strictly sequence model.",
        "mctct": "get_input_embeddings() raises NotImplementedError.",
        "mobilenet_v1": "Not strictly sequence model.",
        "mobilenet_v2": "Not strictly sequence model.",
        "mobilevit": "Not strictly sequence model.",
        "mt5": "Not implemented in EIR for feature extraction yet.",
        "owlvit": "Not strictly sequence model.",
        "nllb": "Throws ValueError: Unrecognized model identifier: nllb",
        "retribert": "Cannot do straightforward look up of embeddings.",
        "resnet": "Not strictly sequence model.",
        "regnet": "Not strictly sequence model.",
        "segformer": "Not strictly sequence model.",
        "sew": "Not strictly sequence model.",
        "sew-d": "Not strictly sequence model.",
        "speech_to_text": "Not strictly sequence model.",
        "swin": "Not strictly sequence model.",
        "swinv2": "Not strictly sequence model.",
        "table-transformer": "Not strictly sequence model.",
        "tapas": "TapasModel requires the torch-scatter library.",
        "time_series_transformer": "Not strictly sequence model.",
        "trajectory_transformer": "Not strictly sequence model.",
        "perceiver": "Not strictly sequence model.",
        "poolformer": "Not strictly sequence model.",
        "qdqbert": "ImportError.",
        "unispeech": "Not strictly sequence model.",
        "unispeech-sat": "Not strictly sequence model.",
        "van": "Not strictly sequence model.",
        "videomae": "Not strictly sequence model.",
        "vilt": "Not strictly sequence model.",
        "vit": "Not strictly sequence model.",
        "vit_mae": "Not strictly sequence model.",
        "vit_msn": "Not strictly sequence model.",
        "vision-text-dual-encoder": "Not strictly sequence model.",
        "xclip": "Not strictly sequence model.",
        "wav2vec2": "Not strictly sequence model.",
        "wav2vec2-conformer": "Not strictly sequence model.",
        "wavlm": "NotImplementedError.",
        "whisper": "Not strictly sequence model.",
        "yolos": "Not strictly sequence model..",
    }

    return unsupported
