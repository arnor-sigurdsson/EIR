from argparse import Namespace
from human_origins_supervised.models import models


def test_make_conv_layers():
    """
    Check that:
        - We have correct number of layers (+2 for first conv layer and
          self attn).
        - We start with correct block (first conv layer).
        - Self attention is the second last layer.

    """
    conv_layer_list = [1, 1, 1, 1]
    test_cl_args = Namespace(
        kernel_width=5,
        down_stride=4,
        target_width=int(8e5),
        rb_do=0.1,
        first_kernel_expansion=1,
        first_stride_expansion=5,
        channel_exp_base=5,
    )
    conv_layers = models.make_conv_layers(conv_layer_list, test_cl_args)

    # account for first block, add +2 instead if using SA
    assert len(conv_layers) == len(conv_layer_list) + 1
    assert isinstance(conv_layers[0], models.FirstBlock)
    # assert isinstance(conv_layers[-2], models.SelfAttention)
