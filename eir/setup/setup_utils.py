from typing import Iterable, Optional, Sequence, Type, Union

import torch
from torch_optimizer import _NAME_OPTIM_MAP
from tqdm import tqdm
from transformers.models.auto.modeling_auto import MODEL_MAPPING_NAMES

from eir.utils.logging import get_logger

al_collector_classes = Union[
    Type["ChannelBasedRunningStatistics"], Type["ElementBasedRunningStatistics"]
]

logger = get_logger(name=__name__)


class ChannelBasedRunningStatistics:
    """
    Adapted from https://gist.github.com/thomasbrandon/ad5b1218fc573c10ea4e1f0c63658469.

    Note assumes CxHxW input shape. final_n_dims is the number of last dimensions
    that are averaged over. For example, if the input shape is (C, H, W) and
    final_n_dims is 2, then the mean and variance will be computed over the
    (H, W) dimensions, resulting in tensors of shape (C,).
    """

    def __init__(self, final_n_dims: int = 2):
        self.final_n_dims: int = final_n_dims
        self.shape: tuple = tuple()
        self.n: int = 0
        self.sum: torch.Tensor = torch.empty(0)
        self.num_var: torch.Tensor = torch.empty(0)

    @torch.no_grad()
    def update(self, data: torch.Tensor):
        data = data.reshape(*list(data.shape[: -self.final_n_dims]) + [-1])

        new_n: int = data.shape[-1]
        new_var: torch.Tensor = data.var(-1)
        new_sum: torch.Tensor = data.sum(-1)

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
    def mean(self) -> torch.Tensor:
        return self.sum / self.n

    @property
    def var(self) -> torch.Tensor:
        return self.num_var / self.n

    @property
    def std(self) -> torch.Tensor:
        return self.var.sqrt()


class ElementBasedRunningStatistics:
    def __init__(self, shape: tuple):
        self.shape = shape
        self.n = 0
        self._mean = torch.zeros(shape)
        self._ssdm = torch.zeros(shape)

    @torch.no_grad()
    def update(self, x: torch.Tensor):
        self.n += 1
        delta = x - self._mean
        self._mean += delta / self.n
        delta2 = x - self._mean
        self._ssdm += delta * delta2

    @property
    def mean(self) -> torch.Tensor:
        return self._mean

    @property
    def var(self) -> torch.Tensor:
        return self._ssdm / (self.n - 1) if self.n > 1 else torch.square(self._mean)

    @property
    def std(self) -> torch.Tensor:
        return torch.sqrt(self.var)


def collect_stats(
    tensor_iterable: Iterable[torch.Tensor],
    collector_class: al_collector_classes,
    shape: tuple,
    max_samples: Optional[int] = None,
    name: Optional[str] = None,
) -> ChannelBasedRunningStatistics | ElementBasedRunningStatistics:
    stats = set_up_collector_instance(collector_class=collector_class, shape=shape)

    for index, it in enumerate(tqdm(tensor_iterable, desc="Gathering Statistics: ")):
        if hasattr(it, "data"):
            stats.update(it.data)
        else:
            stats.update(it)

        if max_samples is not None and index >= max_samples:
            break

    if name is not None:
        logger.info(
            f"Collected stats for {name}, "
            f"with means {stats.mean} and standard deviations {stats.std}."
        )

    return stats


def set_up_collector_instance(
    collector_class: al_collector_classes,
    shape: tuple,
) -> ChannelBasedRunningStatistics | ElementBasedRunningStatistics:
    stats: ChannelBasedRunningStatistics | ElementBasedRunningStatistics
    if collector_class is ChannelBasedRunningStatistics:
        final_n_dims = len(shape) - 1
        stats = ChannelBasedRunningStatistics(final_n_dims=final_n_dims)
    elif collector_class is ElementBasedRunningStatistics:
        stats = ElementBasedRunningStatistics(shape=shape)
    else:
        raise ValueError(
            f"collector_class must be either ChannelBasedRunningStatistics or "
            f"ElementBasedRunningStatistics, "
            f"got {collector_class}."
        )
    return stats


def get_base_optimizer_names() -> set:
    base_names = {"sgdm", "adam", "adamw", "adahessian", "adabelief", "adabeliefw"}

    return base_names


def get_all_optimizer_names() -> Sequence[str]:
    external_optimizers = set(_NAME_OPTIM_MAP.keys())
    base_optimizers = get_base_optimizer_names()
    all_optimizers = set.union(base_optimizers, external_optimizers)
    all_optimizers_list = sorted(list(all_optimizers))

    return all_optimizers_list


def get_all_hf_model_names() -> Sequence[str]:
    all_models = sorted(list(MODEL_MAPPING_NAMES.keys()))
    unsupported = get_unsupported_hf_models()
    unsupported_names = unsupported.keys()
    return [i for i in all_models if i not in unsupported_names]


def get_unsupported_hf_models() -> dict:
    unsupported = {
        "audio-spectrogram-transformer": "Not strictly sequence model.",
        "align": "Not strictly sequence model.",
        "altclip": "Not strictly sequence model.",
        "autoformer": "Not strictly sequence model.",
        "bark": "Not strictly sequence model.",
        "beit": "Not strictly sequence model.",
        "bit": "Not strictly sequence model.",
        "blip": "Not strictly sequence model.",
        "blip-2": "Not strictly sequence model.",
        "bridgetower": "Not strictly sequence model.",
        "bros": "Not strictly sequence model.",
        "canine": "Cannot do straightforward look up of embeddings.",
        "clap": "Not strictly sequence model.",
        "clip": "Not strictly sequence model.",
        "clipseg": "Not strictly sequence model.",
        "clip_vision_model": "Not strictly sequence model.",
        "clvp": "Not strictly sequence model.",
        "chinese_clip": "Not strictly sequence model.",
        "chinese_clip_vision_model": "Not strictly sequence model.",
        "conditional_detr": "Not strictly sequence model.",
        "convbert": "HF error.",
        "convnext": "Not strictly sequence model..",
        "convnextv2": "Not strictly sequence model..",
        "cpmant": "AttributeError: 'NoneType' object has no attribute 'dtype'",
        "cvt": "Not strictly sequence model.",
        "data2vec-audio": "Not strictly sequence model.",
        "data2vec-vision": "Not strictly sequence model.",
        "decision_transformer": "Raises NotImplementedError",
        "decision_transformer_gpt2": "Raises ValueError: Unrecognized model identifier",
        "deformable_detr": "Not strictly sequence model.",
        "deit": "Not strictly sequence model.",
        "deta": "Not strictly sequence model.",
        "dinat": "Not strictly sequence model.",
        "dinov2": "Not strictly sequence model.",
        "detr": "Not strictly sequence model.",
        "donut-swin": "Not strictly sequence model.",
        "dpr": "Not strictly sequence model.",
        "dpt": "Not strictly sequence model.",
        "encodec": "Not strictly sequence model.",
        "esm": "Needs fixing/special handling of init parameters.",
        "efficientformer": "Not strictly sequence model.",
        "efficientnet": "Not strictly sequence model.",
        "ernie_m": "AttributeError: 'bool' object has no attribute 'to'.",
        "fastspeech2_conformer": "Not strictly sequence model.",
        "fsmt": "Not strictly sequence model.",
        "focalnet": "Not strictly sequence model.",
        "funnel": "HF error.",
        "flava": "Not strictly sequence model.",
        "nat": "Requires the natten library",
        "glpn": "Not strictly sequence model.",
        "graphormer": "Not strictly sequence model.",
        "groupvit": "Not strictly sequence model.",
        "grounding-dino": "Not strictly sequence model.",
        "gpt_neo": "Configuration troublesome w.r.t. attn layers matching num_layers.",
        "gptsan-japanese": "AttributeError: 'MoECausalLMOutputWithPast' object "
        "has no attribute 'last_hidden_state'.",
        "hubert": "Cannot do straightforward look up of embeddings.",
        "informer": "Not strictly sequence model.",
        "idefics": "Not strictly sequence model.",
        "idefics2": "Not strictly sequence model.",
        "jamba": "Fast Mamba kernels are not available error.",
        "jukebox": "Not strictly sequence model.",
        "jetmoe": "Shape mismatch error.",
        "kosmos-2": "Not strictly sequence model.",
        "layoutlmv2": "Not strictly sequence model.",
        "layoutlmv3": "Not strictly sequence model.",
        "levit": "Not strictly sequence model.",
        "lilt": "Matmul incompatible shapes with default params.",
        "lxmert": "Not strictly sequence model.",
        "maskformer": "Not strictly sequence model.",
        "mask2former": "Not strictly sequence model.",
        "maskformer-swin": "Not strictly sequence model.",
        "mctct": "get_input_embeddings() raises NotImplementedError.",
        "mgp-str": "Not strictly sequence model.",
        "mistral": "Size mismatch error.",
        "mobilenet_v1": "Not strictly sequence model.",
        "mobilenet_v2": "Not strictly sequence model.",
        "mobilevit": "Not strictly sequence model.",
        "mobilevitv2": "Not strictly sequence model.",
        "musicgen": "Not strictly sequence model.",
        "musicgen_melody": "Not strictly sequence model.",
        "mt5": "Not implemented in EIR for feature extraction yet.",
        "oneformer": "Not strictly sequence model.",
        "open-llama": "'OpenLlamaAttention' object has no attribute 'rope_theta'.",
        "owlvit": "Not strictly sequence model.",
        "owlv2": "Not strictly sequence model.",
        "nllb": "Throws ValueError: Unrecognized model identifier: nllb",
        "retribert": "Cannot do straightforward look up of embeddings.",
        "resnet": "Not strictly sequence model.",
        "regnet": "Not strictly sequence model.",
        "rt_detr": "Not strictly sequence model.",
        "sam": "Not strictly sequence model.",
        "segformer": "Not strictly sequence model.",
        "seamless_m4t": "Not strictly sequence model.",
        "seamless_m4t_v2": "Not strictly sequence model.",
        "sew": "Not strictly sequence model.",
        "sew-d": "Not strictly sequence model.",
        "seggpt": "Not strictly sequence model.",
        "siglip": "Not strictly sequence model.",
        "siglip_vision_model": "Not strictly sequence model.",
        "speech_to_text": "Not strictly sequence model.",
        "speecht5": "Not strictly sequence model.",
        "swiftformer": "Not strictly sequence model.",
        "swin": "Not strictly sequence model.",
        "swinv2": "Not strictly sequence model.",
        "swin2sr": "Not strictly sequence model.",
        "table-transformer": "Not strictly sequence model.",
        "tapas": "TapasModel requires the torch-scatter library.",
        "time_series_transformer": "Not strictly sequence model.",
        "timesformer": "Not strictly sequence model.",
        "timm_backbone": "Not strictly sequence model.",
        "trajectory_transformer": "Not strictly sequence model.",
        "transfo-xl": "Internal TypeError: type_as() missing 1 required positional"
        " arguments: other.",
        "tvlt": "Not strictly sequence model.",
        "tvp": "Not strictly sequence model.",
        "patchtsmixer": "Not strictly sequence model.",
        "patchtst": "Not strictly sequence model.",
        "perceiver": "Not strictly sequence model.",
        "poolformer": "Not strictly sequence model.",
        "pvt": "Not strictly sequence model.",
        "pvt_v2": "Not strictly sequence model.",
        "qdqbert": "ImportError.",
        "udop": "Not strictly sequence model.",
        "unispeech": "Not strictly sequence model.",
        "unispeech-sat": "Not strictly sequence model.",
        "univnet": "Not strictly sequence model.",
        "umt5": "AttributeError: property 'num_hidden_layers' of "
        "'UMT5Config' object has no setter",
        "van": "Not strictly sequence model.",
        "videomae": "Not strictly sequence model.",
        "vilt": "Not strictly sequence model.",
        "vit": "Not strictly sequence model.",
        "vitdet": "Not strictly sequence model.",
        "vits": "Not strictly sequence model.",
        "vivit": "Not strictly sequence model.",
        "vit_hybrid": "Not strictly sequence model.",
        "vit_mae": "Not strictly sequence model.",
        "vit_msn": "Not strictly sequence model.",
        "vision-text-dual-encoder": "Not strictly sequence model.",
        "xclip": "Not strictly sequence model.",
        "xmod": "ValueError: Input language unknown. Please call "
        "`XmodPreTrainedModel.set_default_language().",
        "wav2vec2": "Not strictly sequence model.",
        "wav2vec2-conformer": "Not strictly sequence model.",
        "wav2vec2-bert": "Not strictly sequence model.",
        "wavlm": "NotImplementedError.",
        "whisper": "Not strictly sequence model.",
        "yolos": "Not strictly sequence model..",
    }

    return unsupported
