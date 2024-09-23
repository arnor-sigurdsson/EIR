from eir.models.output.array.array_output_modules import ArrayOutputWrapperModule
from eir.models.output.sequence.sequence_output_modules import SequenceOutputModule
from eir.models.output.tabular.linear import LinearOutputModule
from eir.models.output.tabular.mlp_residual import ResidualMLPOutputModule
from eir.models.output.tabular.shared_mlp_residual import SharedResidualMLPOutputModule

al_output_modules = (
    ResidualMLPOutputModule
    | LinearOutputModule
    | SharedResidualMLPOutputModule
    | SequenceOutputModule
    | ArrayOutputWrapperModule
)
