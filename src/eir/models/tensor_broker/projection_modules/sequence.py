import math
from collections.abc import Callable

import torch
from torch import nn

from eir.models.layers.attention_layers import SwiGLU


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
        target_seq_len: int | None = None,
    ):
        """
        If we are not projecting to a target_seq_length, then the "sequence length"
        is what we get from the reshape function. Otherwise, we account for the
        extra projection in the output_shape and num_out_features properties.
        """
        super().__init__()
        self.input_shape = input_shape_no_batch
        self.target_embedding_dim = target_embedding_dim
        (
            self.reshape_func,
            self.reshaped_out_shape,
        ) = get_reshape_to_attention_dims_func(input_shape=input_shape_no_batch)
        self.direct_reshape_seq_length = self.reshaped_out_shape[0]
        self.target_seq_len = target_seq_len

        self.projection = self._create_projection()

    @property
    def output_shape(self) -> torch.Size:
        if self.target_seq_len is not None:
            return torch.Size([self.target_seq_len, self.target_embedding_dim])
        else:
            return torch.Size(
                [self.direct_reshape_seq_length, self.target_embedding_dim]
            )

    @property
    def num_out_features(self) -> int:
        if self.target_seq_len is not None:
            return self.target_seq_len * self.target_embedding_dim
        else:
            return self.direct_reshape_seq_length * self.target_embedding_dim

    def _create_projection(self):
        n_input_dims = len(self.input_shape)
        target_embedding_dim = self.target_embedding_dim
        if n_input_dims == 1:
            input_embedding_dim = self.input_shape[0]
        elif n_input_dims == 2:
            input_embedding_dim = self.input_shape[1]
        elif n_input_dims == 3:
            # For 3D inputs [c, h, w], use c as the embedding dimension
            input_embedding_dim = self.input_shape[0]
        else:
            input_embedding_dim = math.prod(self.input_shape[1:])

        layers: list[nn.Module]
        layers = [
            nn.Linear(
                in_features=input_embedding_dim,
                out_features=target_embedding_dim,
            ),
        ]

        # While matching the sequence lengths is not strictly necessary,
        # for e.g. cross-attention fusion, it is for example if we
        # are doing e.g. a gated sum, so we keep this sequence length
        # projection here for that case.
        # Note: We don't do this if we already have a 2D shape
        if self.target_seq_len is not None and n_input_dims != 2:
            if n_input_dims == 1:
                linear_layer = nn.Linear(
                    in_features=1,
                    out_features=self.target_seq_len,
                )
            elif n_input_dims == 3:
                # For 3D inputs, sequence length after reshape is h*w
                c, h, w = self.input_shape
                if h * w != self.target_seq_len:
                    linear_layer = nn.Linear(
                        in_features=h * w,
                        out_features=self.target_seq_len,
                    )
                else:
                    return nn.Sequential(*layers)
            elif n_input_dims > 3 and self.input_shape[1] != self.target_seq_len:
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
                    nn.RMSNorm(normalized_shape=target_embedding_dim),
                    SwiGLU(
                        in_features=target_embedding_dim,
                        hidden_features=target_embedding_dim,
                        out_features=target_embedding_dim,
                        bias=False,
                    ),
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
            # 1 because at the point this is called, we have the batch dim
            # hence e.g. [32, 1024] -> [32, 1, 1024]
            return x.unsqueeze(1)

        output_shape = torch.Size([1, input_shape[0]])
    elif n_input_dims == 2:

        def func(x):
            return x

        output_shape = input_shape
    elif n_input_dims == 3:
        c, h, w = input_shape

        def func(x):
            return x.permute(0, 2, 3, 1).reshape(x.size(0), h * w, c)

        output_shape = torch.Size([h * w, c])
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
