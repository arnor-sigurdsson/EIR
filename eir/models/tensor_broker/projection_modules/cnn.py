import itertools

from eir.models.input.array.models_cnn import (
    ConvParamSuggestion,
    conv_output_formula,
    solve_for_padding,
)


def calc_conv_params_for_dimension(
    input_size: int,
    target_size: int,
    min_threshold: float,
    max_kernel_size: int = 7,
    max_stride: int = 4,
    max_dilation: int = 3,
    stride_to_kernel_ratio: float = 1.0,
) -> list[ConvParamSuggestion]:
    if input_size < target_size:
        raise NotImplementedError(
            "When using CNN based projection in tensor broker, "
            "target size currently cannot be larger than input size. "
            "This might be happening if you are sending a tensor message with a "
            "smaller spatial size to a larger one. For this, kindly use a different "
            "projection layer (e.g. 'grouped_linear')."
        )

    valid_params = []

    for kernel_size, stride, dilation in itertools.product(
        range(1, max_kernel_size + 1),
        range(1, max_stride + 1),
        range(1, max_dilation + 1),
    ):
        if stride > kernel_size * stride_to_kernel_ratio:
            continue

        padding = solve_for_padding(
            input_size=input_size,
            target_size=target_size,
            dilation=dilation,
            stride=stride,
            kernel_size=kernel_size,
        )

        if padding is None:
            continue

        output_size = conv_output_formula(
            input_size=input_size,
            padding=padding,
            dilation=dilation,
            kernel_size=kernel_size,
            stride=stride,
        )

        size_ratio = output_size / target_size

        if size_ratio >= min_threshold:
            valid_params.append(
                ConvParamSuggestion(
                    kernel_size=kernel_size,
                    stride=stride,
                    dilation=dilation,
                    padding=padding,
                    target_size=output_size,
                )
            )

    if not valid_params:
        raise ValueError(
            f"No valid convolutional parameters found to transform "
            f"{input_size} to {target_size} with min_threshold {min_threshold}"
        )

    return valid_params


def choose_best_params(
    params: list[ConvParamSuggestion], target_size: int
) -> ConvParamSuggestion:
    """
    Slightly confusing here that the output size of the param suggestion is
    named target_size, but it is actually the output size of the convolution.
    """
    return min(
        params,
        key=lambda p: (
            abs(p.target_size - target_size),
            p.kernel_size + p.stride + p.dilation + p.padding,
        ),
    )


def get_conv_params_for_dimension(
    input_size: int,
    target_size: int,
    min_threshold: float = 1.0,
    max_kernel_size: int = 33,
    max_stride: int = 32,
    max_dilation: int = 4,
) -> ConvParamSuggestion:
    valid_params = calc_conv_params_for_dimension(
        input_size=input_size,
        target_size=target_size,
        min_threshold=min_threshold,
        max_kernel_size=max_kernel_size,
        max_stride=max_stride,
        max_dilation=max_dilation,
    )
    best_solution = choose_best_params(
        params=valid_params,
        target_size=target_size,
    )

    return best_solution
