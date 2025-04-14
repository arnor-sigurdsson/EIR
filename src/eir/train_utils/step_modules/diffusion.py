from dataclasses import dataclass

import torch
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler
from torch.nn import functional as F


@dataclass
class DiffusionConfig:
    time_steps: int
    betas: torch.Tensor
    sqrt_alphas_cumprod: torch.Tensor
    sqrt_one_minus_alphas_cumprod: torch.Tensor
    sqrt_recip_alphas: torch.Tensor
    posterior_variance: torch.Tensor


def initialize_diffusion_config(time_steps: int, beta_schedule: str) -> DiffusionConfig:
    """
    Note that DDPMScheduler, DDIM, and other schedulers are all using the
    same core function to compute the actual betas, hence does not matter
    which one we use here to initialize the betas and other parameters.
    """

    scheduler = DDPMScheduler(
        num_train_timesteps=time_steps,
        beta_schedule=beta_schedule,
        prediction_type="v_prediction",
        rescale_betas_zero_snr=True,
    )

    betas = scheduler.betas
    alphas = 1.0 - betas
    alphas_cumprod = scheduler.alphas_cumprod
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

    sqrt_alphas_cumprod_t = extract(
        a=diffusion_config.sqrt_alphas_cumprod.to(device=device),
        t=t,
        x_shape=inputs.shape,
    )
    sqrt_one_minus_alphas_cumprod_t = extract(
        a=diffusion_config.sqrt_one_minus_alphas_cumprod.to(device=device),
        t=t,
        x_shape=inputs.shape,
    )

    v_target = sqrt_alphas_cumprod_t * noise - sqrt_one_minus_alphas_cumprod_t * inputs

    return x_noisy, v_target, t


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

    sqrt_alphas_cumprod_t = sqrt_alphas_cumprod_t.to(device=device)
    sqrt_one_minus_alphas_cumprod_t = sqrt_one_minus_alphas_cumprod_t.to(device=device)
    scaled_x_start = scaled_x_start.to(device=device)
    noise = noise.to(device=device)

    x_noisy = (
        sqrt_alphas_cumprod_t * scaled_x_start + sqrt_one_minus_alphas_cumprod_t * noise
    )

    return x_noisy


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
