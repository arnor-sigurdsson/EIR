from typing import TYPE_CHECKING, DefaultDict

if TYPE_CHECKING:
    from eir.data_load.datasets import Sample


def add_id_to_samples(
    samples: DefaultDict[str, "Sample"], sample_id: str
) -> DefaultDict[str, "Sample"]:
    """
    This kind of weird function is used because in some cases, we cannot expect the
    target labels to have added samples, because we could be predicting on completely
    unknown samples without any target label data.

    Hence, we might have sparse modular data available for the samples, e.g. only omics
    for some samples, but only tabular data for others. So we want to ensure that the
    data is filled in.
    """
    if not samples[sample_id].sample_id:
        samples[sample_id].sample_id = sample_id
    else:
        assert samples[sample_id].sample_id == sample_id

    return samples
