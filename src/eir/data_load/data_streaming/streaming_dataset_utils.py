from eir.setup.input_setup_modules.setup_sequence import get_sequence_split_function


def streamline_sequence_manual_data(data: str, split_on: str | None) -> list[str] | str:
    """
    This is to specifically handle the case of an empty string / None being passed
    here. If e.g. we call the split_func on '', we will get [''], which will
    end up being encoded as a <unk> token. Instead, we want to return an empty
    list here. In e.g. the validation handler code, this is also set explicitly.
    """

    sequence_streamlined: list[str] | str
    if data == "" or data is None:
        sequence_streamlined = []
    else:
        split_func = get_sequence_split_function(split_on=split_on)
        split_data = split_func(data)
        sequence_streamlined = split_data

    return sequence_streamlined
