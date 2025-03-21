from dataclasses import dataclass

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


@dataclass
class DiffusionConfig:
    time_steps: int
    betas: torch.Tensor
    sqrt_alphas_cumprod: torch.Tensor
    sqrt_one_minus_alphas_cumprod: torch.Tensor
    sqrt_recip_alphas: torch.Tensor
    posterior_variance: torch.Tensor


def initialize_diffusion_config(time_steps: int) -> DiffusionConfig:
    beta_start = 0.0001
    beta_end = 0.02
    betas = torch.linspace(beta_start, beta_end, time_steps)

    alphas = 1.0 - betas
    alphas_cumprod = torch.cumprod(alphas, dim=0)
    alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value=1.0)

    sqrt_recip_alphas = torch.sqrt(1.0 / alphas)
    sqrt_alphas_cumprod = torch.sqrt(alphas_cumprod)
    sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - alphas_cumprod)
    posterior_variance = betas * (1.0 - alphas_cumprod_prev) / (1.0 - alphas_cumprod)

    return DiffusionConfig(
        time_steps=time_steps,
        betas=betas,
        sqrt_alphas_cumprod=sqrt_alphas_cumprod,
        sqrt_one_minus_alphas_cumprod=sqrt_one_minus_alphas_cumprod,
        sqrt_recip_alphas=sqrt_recip_alphas,
        posterior_variance=posterior_variance,
    )


def prepare_diffusion_batch(
    diffusion_config: DiffusionConfig,
    inputs: torch.Tensor,
    batch_size: int,
    num_steps: int,
    device: str,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    t = torch.randint(
        low=0,
        high=num_steps,
        size=(batch_size,),
        dtype=torch.long,
        device=device,
    )

    noise = torch.randn_like(inputs, device=device)

    x_noisy = q_sample(
        config=diffusion_config,
        x_start=inputs,
        t=t,
        noise=noise,
        device=device,
    )

    return x_noisy, noise, t


def q_sample(
    config: DiffusionConfig,
    x_start: torch.Tensor,
    t: torch.Tensor,
    noise: torch.Tensor,
    device: str,
    input_scale: float = 1.0,
) -> torch.Tensor:
    """
    This and other functions adapted from
    https://huggingface.co/blog/annotated-diffusion
    """

    sqrt_alphas_cumprod_t = extract(
        a=config.sqrt_alphas_cumprod.to(device=device),
        t=t.to(device=device),
        x_shape=x_start.shape,
    )

    sqrt_one_minus_alphas_cumprod_t = extract(
        a=config.sqrt_one_minus_alphas_cumprod.to(device=device),
        t=t.to(device=device),
        x_shape=x_start.shape,
    )

    scaled_x_start = input_scale * x_start

    sqrt_alphas_cumprod_t = sqrt_alphas_cumprod_t.to(device)
    sqrt_one_minus_alphas_cumprod_t = sqrt_one_minus_alphas_cumprod_t.to(device)
    scaled_x_start = scaled_x_start.to(device)
    noise = noise.to(device)

    x_noisy = (
        sqrt_alphas_cumprod_t * scaled_x_start + sqrt_one_minus_alphas_cumprod_t * noise
    )

    return x_noisy


@torch.no_grad()
def p_sample_loop(
    config: DiffusionConfig,
    batch_inputs: dict,
    output_name: str,
    model: nn.Module,
    output_shape: tuple,
    time_steps: int,
    device: str,
) -> np.ndarray:
    current_state: torch.Tensor = torch.randn(output_shape, device=device)
    batch_inputs[output_name] = current_state

    batch_size = output_shape[0]

    for i in reversed(range(0, time_steps)):
        t = torch.full(
            size=(batch_size,),
            fill_value=i,
            dtype=torch.long,
            device=device,
        )

        current_state = p_sample(
            config=config,
            model=model,
            output_name=output_name,
            batch_inputs=batch_inputs,
            t=t,
            t_index=i,
            device=device,
        )

        batch_inputs[output_name] = current_state

    return current_state.cpu().numpy()


@torch.no_grad()
def p_sample(
    config: DiffusionConfig,
    model: nn.Module,
    batch_inputs: dict[str, torch.Tensor],
    output_name: str,
    t: torch.Tensor,
    t_index: int,
    device: str,
) -> torch.Tensor:
    """
    TODO: Move all config tensors to device beforehand.
    """

    current_state = batch_inputs[output_name].to(device)
    t = t.to(device)

    output_module = getattr(model.output_modules, output_name)
    t_emb = output_module.feature_extractor.timestep_embeddings(t)
    batch_inputs[f"__extras_{output_name}"] = t_emb

    betas_t = extract(
        a=config.betas.to(device),
        t=t,
        x_shape=current_state.shape,
    )
    sqrt_one_minus_alphas_cumprod_t = extract(
        a=config.sqrt_one_minus_alphas_cumprod.to(device),
        t=t,
        x_shape=current_state.shape,
    )
    sqrt_recip_alphas_t = extract(
        a=config.sqrt_recip_alphas.to(device),
        t=t,
        x_shape=current_state.shape,
    )

    model_outputs = model(batch_inputs)
    model_diffusion_output = model_outputs[output_name][output_name]

    model_mean = sqrt_recip_alphas_t * (
        current_state
        - betas_t * model_diffusion_output / sqrt_one_minus_alphas_cumprod_t
    )

    if t_index == 0:
        return model_mean

    posterior_variance_t = extract(
        a=config.posterior_variance.to(device),
        t=t,
        x_shape=current_state.shape,
    )
    noise = torch.randn_like(current_state)
    return model_mean + torch.sqrt(posterior_variance_t) * noise


def extract(
    a: torch.Tensor,
    t: torch.Tensor,
    x_shape: tuple,
) -> torch.Tensor:
    """
    Allows us to extract the appropriate index for a batch of indices.
    """
    batch_size = t.shape[0]
    out = a.gather(-1, t)
    return out.reshape(batch_size, *((1,) * (len(x_shape) - 1))).to(t.device)
