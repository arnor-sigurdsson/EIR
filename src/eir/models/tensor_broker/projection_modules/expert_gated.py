import torch
from torch import nn

from eir.models.layers.mlp_layers import MLPResidualBlock
from eir.models.layers.projection_layers import get_1d_projection_layer
from eir.setup.schema_modules.tensor_broker_schemas import al_broker_projection_types


class ExpertGatedProjection(nn.Module):
    def __init__(
        self,
        num_experts: int,
        expert_dim: int,
        target_dim: int,
        projection_type: al_broker_projection_types = "lcl+mlp_residual",
        kernel_width_divisible_by: int | None = None,
        projection_intermediate_factor: int | None = None,
    ):
        super().__init__()

        self.num_experts = num_experts
        self.expert_dim = expert_dim
        self.target_dim = target_dim

        self.expert_gate = nn.Parameter(torch.zeros(num_experts))

        self.projection = _build_projection(
            input_dim=expert_dim,
            target_dim=target_dim,
            projection_type=projection_type,
            kernel_width_divisible_by=kernel_width_divisible_by,
            projection_intermediate_factor=projection_intermediate_factor,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        expert_slices = x.reshape(x.shape[0], self.num_experts, self.expert_dim)

        gate_weights = torch.softmax(self.expert_gate, dim=0)
        gated = (gate_weights.unsqueeze(0).unsqueeze(-1) * expert_slices).sum(dim=1)

        return self.projection(gated)


def _build_projection(
    input_dim: int,
    target_dim: int,
    projection_type: al_broker_projection_types,
    kernel_width_divisible_by: int | None,
    projection_intermediate_factor: int | None,
) -> nn.Module:
    layers: list[nn.Module] = []

    match projection_type:
        case "lcl" | "linear" | "lcl_residual":
            projection = get_1d_projection_layer(
                input_dimension=input_dim,
                target_dimension=target_dim,
                projection_layer_type=projection_type,
                lcl_diff_tolerance=0,
                kernel_width_divisible_by=kernel_width_divisible_by,
            )
            layers.append(projection)

        case "lcl+mlp_residual":
            factor = projection_intermediate_factor or 1
            lcl_target_dim = target_dim * factor

            layers.append(nn.RMSNorm(normalized_shape=input_dim))
            layers.append(nn.GELU())
            layers.append(
                get_1d_projection_layer(
                    input_dimension=input_dim,
                    target_dimension=lcl_target_dim,
                    projection_layer_type="lcl",
                    lcl_diff_tolerance=0,
                    kernel_width_divisible_by=kernel_width_divisible_by,
                )
            )
            layers.append(
                MLPResidualBlock(
                    in_features=lcl_target_dim,
                    out_features=target_dim,
                    dropout_p=0.0,
                    full_preactivation=True,
                    stochastic_depth_p=0.0,
                )
            )

        case "mlp_residual":
            layers.append(
                MLPResidualBlock(
                    in_features=input_dim,
                    out_features=target_dim,
                    dropout_p=0.0,
                    full_preactivation=True,
                    stochastic_depth_p=0.0,
                )
            )

        case _:
            raise ValueError(
                f"Projection type '{projection_type}' is not supported "
                f"for expert-gated projection. "
                f"Supported types: lcl, linear, lcl_residual, "
                f"lcl+mlp_residual, mlp_residual."
            )

    return nn.Sequential(*layers)
