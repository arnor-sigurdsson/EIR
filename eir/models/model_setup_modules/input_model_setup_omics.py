from eir.models.input.omics.omics_models import (
    al_omics_model_configs,
    get_omics_model_class,
    al_omics_model_types,
    get_omics_model_init_kwargs,
)
from eir.setup.input_setup_modules.common import DataDimensions


def get_omics_model_from_model_config(
    model_init_config: al_omics_model_configs,
    data_dimensions: DataDimensions,
    model_type: al_omics_model_types,
):
    omics_model_class = get_omics_model_class(model_type=model_type)
    model_init_kwargs = get_omics_model_init_kwargs(
        model_type=model_type,
        model_config=model_init_config,
        data_dimensions=data_dimensions,
    )
    omics_model = omics_model_class(**model_init_kwargs)

    if model_type == "cnn":
        assert omics_model.data_size_after_conv >= 8

    return omics_model
