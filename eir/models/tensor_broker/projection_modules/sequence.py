import math
from typing import Callable, Optional

import torch
from torch import nn


class Transpose(nn.Module):
    def __init__(self, dim0, dim1):
        super().__init__()
        self.dim0 = dim0
        self.dim1 = dim1

    def forward(self, x):
        return x.transpose(self.dim0, self.dim1)


class SequenceProjectionLayer(nn.Module):
    def __init__(
        self,
        input_shape_no_batch: torch.Size,
        target_embedding_dim: int,
        target_seq_len: Optional[int] = None,
    ):
        super().__init__()
        self.input_shape = input_shape_no_batch
        self.target_embedding_dim = target_embedding_dim
        self.target_seq_len = target_seq_len
        self.reshape_func, _ = get_reshape_to_attention_dims_func(
            input_shape=input_shape_no_batch
        )
        self.projection = self._create_projection()

    def _create_projection(self):
        n_input_dims = len(self.input_shape)
        target_embedding_dim = self.target_embedding_dim
        if n_input_dims == 1:
            input_embedding_dim = self.input_shape[0]
        elif n_input_dims == 2:
            input_embedding_dim = self.input_shape[1]
        else:
            input_embedding_dim = math.prod(self.input_shape[1:])

        layers: list[nn.Module]
        layers = [
            nn.Linear(
                in_features=input_embedding_dim,
                out_features=target_embedding_dim,
            ),
        ]

        if self.target_seq_len is not None:
            if n_input_dims == 2:
                linear_layer = nn.Linear(
                    in_features=1,
                    out_features=self.target_seq_len,
                )
            elif n_input_dims > 2 and self.input_shape[1] != self.target_seq_len:
                linear_layer = nn.Linear(
                    in_features=self.input_shape[1],
                    out_features=self.target_seq_len,
                )
            else:
                return nn.Sequential(*layers)

            # here operating with batch dim, hence offset by 1 compared to the logic
            # above to get 1 and 2
            layers.extend(
                [
                    nn.LayerNorm(normalized_shape=target_embedding_dim),
                    nn.GELU(),
                    Transpose(1, 2),
                    linear_layer,
                    Transpose(1, 2),
                ]
            )

        return nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.reshape_func(x)
        out = self.projection(out)
        return out


def get_reshape_to_attention_dims_func(
    input_shape: torch.Size,
) -> tuple[Callable[[torch.Tensor], torch.Tensor], torch.Size]:
    n_input_dims = len(input_shape)

    if n_input_dims == 1:

        def func(x):
            return x.unsqueeze(0)

        output_shape = torch.Size([1, input_shape[0]])
    elif n_input_dims == 2:

        def func(x):
            return x

        output_shape = input_shape
    else:

        def func(x):
            """
            Note here start at 2 since we have the batch dim when this is called.
            """
            return x.flatten(start_dim=2)

        output_shape = torch.Size(
            [input_shape[0], int(torch.prod(torch.tensor(input_shape[1:])).item())]
        )

    return func, output_shape
