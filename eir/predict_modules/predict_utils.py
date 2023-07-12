from typing import Dict, Sequence, Tuple

import colorama as clr


def log_and_raise_missing_or_multiple_config_matching_general(
    train_name: str,
    train_type: str,
    matches: Sequence[Dict],
    predict_names_and_types: Sequence[Tuple[str, str]],
    name: str,
) -> None:
    assert name in ("input", "output")
    train_name = clr.Style.BRIGHT + clr.Fore.GREEN + train_name + clr.Style.RESET_ALL
    train_type = clr.Style.BRIGHT + clr.Fore.GREEN + train_type + clr.Style.RESET_ALL

    predict_names_and_types_formatted = ""
    for name, type_ in predict_names_and_types:
        name = clr.Style.BRIGHT + clr.Fore.RED + name + clr.Style.RESET_ALL
        type_ = clr.Style.BRIGHT + clr.Fore.RED + type_ + clr.Style.RESET_ALL
        predict_names_and_types_formatted += f"('{name}', '{type_}'), "
    predict_names_and_types_formatted = predict_names_and_types_formatted[:-2]

    if len(matches) == 0:
        msg = (
            f"When running the predict module, no *predict* {name} config was found "
            f"that matches a configuration used during training. There must be "
            f"exactly one match for each {name} config used during training. "
            f"The following {name} configs were used during training and no match "
            f"was found (name, type): ('{train_name}', '{train_type}'). "
            f"The following {name} configs were found in the *predict* config file(s): "
            f"{predict_names_and_types_formatted}. "
            f"To fix this, make sure that the {name} configs passed to the predict "
            f"module match the {name} configs used during training."
        )
    else:
        msg = (
            f"When running the predict module, multiple *predict* {name} configs were "
            f"found that match a configuration used during training. There must be "
            f"exactly one match for each {name} config used during training. "
            f"The following {name} configs were used during training and multiple "
            f"was found (name, type): ('{train_name}', '{train_type}'). "
            f"The following {name} configs were found in the *predict* config file(s): "
            f"{predict_names_and_types_formatted}. "
            f"To fix this, make sure that the {name} configs passed to the predict "
            f"module match the {name} configs used during training."
        )

    msg = msg.replace(
        "*predict*", clr.Style.BRIGHT + clr.Fore.RED + "*predict*" + clr.Style.RESET_ALL
    )

    raise ValueError(msg)


def log_and_raise_missing_or_multiple_tabular_output_matches(
    train_name: str,
    train_type: str,
    train_cat_columns: Sequence[str],
    train_con_columns: Sequence[str],
    matches: Sequence[Dict],
    predict_names_and_types: Sequence[Tuple[str, str, Sequence[str], Sequence[str]]],
) -> None:
    train_name = clr.Style.BRIGHT + clr.Fore.GREEN + train_name + clr.Style.RESET_ALL
    train_type = clr.Style.BRIGHT + clr.Fore.GREEN + train_type + clr.Style.RESET_ALL
    train_cat_columns = (
        clr.Style.BRIGHT + clr.Fore.GREEN + str(train_cat_columns) + clr.Style.RESET_ALL
    )
    train_con_columns = (
        clr.Style.BRIGHT + clr.Fore.GREEN + str(train_con_columns) + clr.Style.RESET_ALL
    )

    predict_names_and_types_formatted = ""
    for name, type_, cat_cols, con_cols in predict_names_and_types:
        name = clr.Style.BRIGHT + clr.Fore.RED + name + clr.Style.RESET_ALL
        type_ = clr.Style.BRIGHT + clr.Fore.RED + type_ + clr.Style.RESET_ALL
        cat_cols = clr.Style.BRIGHT + clr.Fore.RED + str(cat_cols) + clr.Style.RESET_ALL
        con_cols = clr.Style.BRIGHT + clr.Fore.RED + str(con_cols) + clr.Style.RESET_ALL

        predict_names_and_types_formatted += (
            f"('{name}', '{type_}', cat_cols: '{cat_cols}', con_cols: '{con_cols}'), "
        )
    predict_names_and_types_formatted = predict_names_and_types_formatted[:-2]

    if len(matches) == 0:
        msg = (
            f"When running the predict module, no *predict* output config was found "
            f"that matches a configuration used during training. There must be "
            f"exactly one match for each output config used during training. "
            f"The following output configs were used during training and no match "
            f"was found (name, type): ('{train_name}', '{train_type}') "
            f"with target columns "
            f"cat: {train_cat_columns} and con: {train_con_columns}. "
            f"The following output configs were found in the *predict* config file(s): "
            f"{predict_names_and_types_formatted}. "
            f"To fix this, make sure that the output configs passed to the predict "
            f"module match the output configs used during training."
        )
    else:
        msg = (
            f"When running the predict module, multiple *predict* output configs were "
            f"found that match a configuration used during training. There must be "
            f"exactly one match for each output config used during training. "
            f"The following output configs were used during training and multiple "
            f"was found (name, type): ('{train_name}', '{train_type}') "
            f"with target columns "
            f"cat: {train_cat_columns} and con: {train_con_columns}. "
            f"The following output configs were found in the *predict* config file(s): "
            f"{predict_names_and_types_formatted}. "
            f"To fix this, make sure that the output configs passed to the predict "
            f"module match the output configs used during training."
        )

    msg = msg.replace(
        "*predict*", clr.Style.BRIGHT + clr.Fore.RED + "*predict*" + clr.Style.RESET_ALL
    )
    raise ValueError(msg)
