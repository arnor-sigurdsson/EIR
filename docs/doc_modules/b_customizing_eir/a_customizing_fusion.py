from pathlib import Path

import torch
from torch import nn

import eir.models.models_utils
from eir import train
from eir.models.fusion.fusion import al_fused_features
from eir.models.meta.meta import MetaModel
from eir.models.meta.meta_utils import FeatureExtractorProtocol
from eir.models.model_setup_modules.meta_setup import get_output_modules
from eir.setup.config import get_configs
from eir.train_utils import step_logic
from eir.train_utils.utils import configure_global_eir_logging


def main():
    configs = get_configs()

    configure_global_eir_logging(output_folder=configs.gc.be.output_folder)

    default_hooks = step_logic.get_default_hooks(configs=configs)
    default_experiment = train.get_default_experiment(
        configs=configs,
        hooks=default_hooks,
    )

    my_experiment = modify_experiment(experiment=default_experiment)

    train.run_experiment(experiment=my_experiment)


class MyLSTMFusionModule(nn.Module):
    def __init__(self, fusion_in_dim: int, out_dim: int):
        """
        An example of a custom fusion module. Here we use a simple LSTM to
        fuse the inputs, but you could use any PyTorch module here.
        """
        super().__init__()

        self.fusion_in_dim = fusion_in_dim
        self.out_dim = out_dim

        self.fusion = nn.LSTM(
            input_size=fusion_in_dim,
            hidden_size=self.out_dim,
            num_layers=1,
            batch_first=True,
        )

    @property
    def num_out_features(self) -> int:
        return self.out_dim

    @property
    def output_shape(self) -> tuple[int, ...]:
        return (self.out_dim,)

    def forward(self, inputs: dict[str, FeatureExtractorProtocol]) -> al_fused_features:
        features = torch.cat(tuple(inputs.values()), dim=1)
        assert features.shape[1] == self.fusion_in_dim

        out, *_ = self.fusion(features)

        return out


def modify_experiment(experiment: train.Experiment) -> train.Experiment:
    my_experiment_attributes = experiment.__dict__

    input_modules = experiment.model.input_modules
    fusion_in_dim = sum(i.num_out_features for i in input_modules.values())

    my_fusion_module = MyLSTMFusionModule(fusion_in_dim=fusion_in_dim, out_dim=128)
    my_fusion_modules = nn.ModuleDict({"computed": my_fusion_module})

    my_output_modules, _ = get_output_modules(
        outputs_as_dict=experiment.outputs,
        computed_out_dimensions=my_fusion_module.num_out_features,
        device=experiment.configs.global_config.basic_experiment.device,
        fusion_model_type="computed",
    )

    my_model = MetaModel(
        input_modules=input_modules,
        fusion_modules=my_fusion_modules,
        output_modules=my_output_modules,
        fusion_to_output_mapping={"ancestry_output": "computed"},
        tensor_broker=nn.ModuleDict(),
    )

    my_optimizer = torch.optim.Adam(
        params=my_model.parameters(),
        lr=1e-4,
    )

    fabric = train.setup_accelerator(configs=experiment.configs)

    my_model, my_optimizer = fabric.setup(my_model, my_optimizer)

    my_experiment_attributes["model"] = my_model
    my_experiment_attributes["optimizer"] = my_optimizer

    my_experiment = train.Experiment(**my_experiment_attributes)

    output_folder = my_experiment.configs.global_config.be.output_folder
    eir.models.models_utils.log_model(
        model=my_model,
        structure_file=Path(output_folder, "model_info.txt"),
    )

    return my_experiment


if __name__ == "__main__":
    main()
