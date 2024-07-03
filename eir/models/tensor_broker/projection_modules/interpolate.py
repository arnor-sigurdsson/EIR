from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class InterpolateProjection(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        output_size: Tuple[int, int],
        mode: str = "bilinear",
        align_corners: Optional[bool] = None,
    ):
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.output_size = output_size
        self.mode = mode
        self.align_corners = align_corners

        self.sampling = ArbitrarySampling(
            output_size=output_size,
            mode=mode,
            align_corners=align_corners,
        )

        self.norm = nn.GroupNorm(num_groups=1, num_channels=self.out_channels)

        self.conv_1 = nn.Conv2d(
            in_channels=self.out_channels,
            out_channels=self.out_channels,
            kernel_size=3,
            stride=1,
            padding=1,
        )

        self.act = nn.GELU()

        self.downsample_identity: nn.Module
        if in_channels != out_channels:
            self.downsample_identity = nn.Conv2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=1,
                stride=1,
                padding=0,
            )
        else:
            self.downsample_identity = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = self.downsample_identity(x)
        identity = self.sampling(identity)

        out = self.norm(identity)
        out = self.conv_1(out)
        out = self.act(out)

        return out + identity


class ArbitrarySampling(nn.Module):
    def __init__(
        self,
        output_size: Tuple[int, int],
        mode: str = "bilinear",
        align_corners: Optional[bool] = None,
    ):
        super().__init__()

        self.output_size = output_size
        self.mode = mode
        self.align_corners = align_corners

    def __repr__(self):
        return (
            f"{self.__class__.__name__}(output_size={self.output_size}, "
            f"mode={self.mode}, align_corners={self.align_corners})"
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.interpolate(
            input=x,
            size=self.output_size,
            mode=self.mode,
            align_corners=self.align_corners,
        )
