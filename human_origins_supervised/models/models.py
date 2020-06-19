from argparse import Namespace
from collections import OrderedDict
from typing import List, Union, Tuple, Dict, Callable

import torch
from aislib import pytorch_utils
from aislib.misc_utils import get_logger
from aislib.pytorch_modules import Swish
from torch import nn

from human_origins_supervised.data_load.datasets import al_num_classes
from . import extra_inputs_module
from .extra_inputs_module import al_emb_lookup_dict
from .layers import SelfAttention, FirstBlock, Block, SplitLinear
from .model_utils import find_no_resblocks_needed

# type aliases
al_models = Union["CNNModel", "MLPModel", "LinearModel"]

logger = get_logger(name=__name__, tqdm_compatible=True)


def get_model_class(model_type: str) -> al_models:
    if model_type == "cnn":
        return CNNModel
    elif model_type == "mlp":
        return MLPModel
    elif model_type == "mlp-split":
        return SplitMLPModel
    elif model_type == "mlp-mgmoe":
        return MGMoEModel

    return LinearModel


def _get_block(
    conv_blocks: List[nn.Module],
    layer_arch_idx: int,
    down_stride: int,
    cl_args: Namespace,
) -> Tuple[Block, int]:
    ca = cl_args

    cur_conv = nn.Sequential(*conv_blocks)
    cur_width = pytorch_utils.calc_size_after_conv_sequence(
        input_width=ca.target_width, conv_sequence=cur_conv
    )

    cur_kern, cur_padd = pytorch_utils.calc_conv_params_needed(
        input_width=cur_width,
        kernel_size=ca.kernel_width,
        stride=down_stride,
        dilation=1,
    )

    cur_block_number = len([i for i in conv_blocks if isinstance(i, Block)]) + 1
    cur_dilation_factor = _get_cur_dilation(
        dilation_factor=cl_args.dilation_factor,
        width=cur_width,
        block_number=cur_block_number,
    )

    cur_in_channels = conv_blocks[-1].out_channels
    cur_out_channels = 2 ** (ca.channel_exp_base + layer_arch_idx)

    cur_layer = Block(
        in_channels=cur_in_channels,
        out_channels=cur_out_channels,
        conv_1_kernel_w=cur_kern,
        conv_1_padding=cur_padd,
        down_stride_w=down_stride,
        dilation=cur_dilation_factor,
        full_preact=True if len(conv_blocks) == 1 else False,
        rb_do=ca.rb_do,
    )

    return cur_layer, cur_width


def _get_cur_dilation(dilation_factor: int, width: int, block_number: int):
    """
    Note that block_number refers to the number of residual blocks (not first block
    or self attention).
    """
    dilation = dilation_factor ** block_number

    while dilation >= width:
        dilation = dilation // dilation_factor

    return dilation


def _make_conv_layers(
    residual_blocks: List[int], cl_args: Namespace
) -> List[nn.Module]:
    """
    Used to set up the convolutional layers for the model. Based on the passed in
    residual blocks, we want to set up the actual blocks with all the relevant
    convolution parameters.

    We start with a base channel number of 2**5 == 32.

    Also inserts a self-attention layer in just before the last residual block.

    :param residual_blocks: List of ints, where each int indicates number of blocks w.
    that channel dimension.
    :param cl_args: Experiment hyperparameters / configuration needed for the
    convolution setup.
    :return: A list of `nn.Module` objects to be passed to `nn.Sequential`.
    """
    ca = cl_args

    down_stride_w = ca.down_stride

    first_conv_channels = 2 ** ca.channel_exp_base * ca.first_channel_expansion
    first_conv_kernel = ca.kernel_width * ca.first_kernel_expansion
    first_conv_stride = down_stride_w * ca.first_stride_expansion

    first_kernel, first_pad = pytorch_utils.calc_conv_params_needed(
        input_width=ca.target_width,
        kernel_size=first_conv_kernel,
        stride=first_conv_stride,
        dilation=1,
    )

    conv_blocks = [
        FirstBlock(
            in_channels=1,
            out_channels=first_conv_channels,
            conv_1_kernel_w=first_kernel,
            conv_1_padding=first_pad,
            down_stride_w=first_conv_stride,
            dilation=1,
            rb_do=ca.rb_do,
        )
    ]

    sa_added = False
    for layer_arch_idx, layer_arch_layers in enumerate(residual_blocks):
        for layer in range(layer_arch_layers):
            cur_layer, cur_width = _get_block(
                conv_blocks=conv_blocks,
                layer_arch_idx=layer_arch_idx,
                down_stride=down_stride_w,
                cl_args=ca,
            )

            if cl_args.sa and cur_width < 1024 and not sa_added:
                attention_channels = conv_blocks[-1].out_channels
                conv_blocks.append(SelfAttention(attention_channels))
                sa_added = True

            conv_blocks.append(cur_layer)

    return conv_blocks


class ModelBase(nn.Module):
    def __init__(
        self,
        cl_args: Namespace,
        num_classes: al_num_classes,
        embeddings_dict: Union[al_emb_lookup_dict, None] = None,
        extra_continuous_inputs_columns: Union[List[str], None] = None,
    ):
        super().__init__()

        self.cl_args = cl_args
        self.num_classes = num_classes
        self.embeddings_dict = embeddings_dict
        self.extra_continuous_inputs_columns = extra_continuous_inputs_columns

        emb_total_dim = con_total_dim = 0
        if embeddings_dict:
            emb_total_dim = extra_inputs_module.attach_embeddings(self, embeddings_dict)
        if extra_continuous_inputs_columns:
            con_total_dim = len(self.extra_continuous_inputs_columns)

        self.fc_repr_and_extra_dim = cl_args.fc_repr_dim
        self.fc_task_dim = cl_args.fc_task_dim

        # TODO: Better to have this a method so fc_extra is explicitly defined?
        self.extra_dim = emb_total_dim + con_total_dim
        if emb_total_dim or con_total_dim:
            # we have a specific layer for fc_extra in case it's going straight
            # to bn or act, ensuring linear before
            self.fc_extra = nn.Linear(self.extra_dim, self.extra_dim, bias=False)
            self.fc_repr_and_extra_dim += self.extra_dim

    @property
    def fc_1_in_features(self) -> int:
        raise NotImplementedError


def _get_module_dict_from_target_columns(num_classes: al_num_classes, fc_in: int):

    module_dict = {}
    for key, num_classes in num_classes.items():
        module_dict[key] = nn.Linear(fc_in, num_classes)

    return nn.ModuleDict(module_dict)


def _get_multi_task_branches(
    fc_repr_and_extra_dim: int,
    fc_task_dim: int,
    fc_do: float,
    num_classes: al_num_classes,
) -> nn.ModuleDict:

    module_dict = {}
    for key, num_classes in num_classes.items():
        branch_layers = OrderedDict(
            {
                "fc_2_bn_1": nn.BatchNorm1d(fc_repr_and_extra_dim),
                "fc_2_act_1": Swish(),
                "fc_2_linear_1": nn.Linear(
                    fc_repr_and_extra_dim, fc_task_dim, bias=False
                ),
                "fc_2_do_1": nn.Dropout(p=fc_do),
                "fc_3_bn_1": nn.BatchNorm1d(fc_task_dim),
                "fc_3_act_1": Swish(),
                "fc_3_do_1": nn.Dropout(p=fc_do),
            }
        )

        task_layer_branch = nn.Sequential(
            OrderedDict(
                **branch_layers, **{"fc_3_final": nn.Linear(fc_task_dim, num_classes)}
            )
        )

        module_dict[key] = task_layer_branch

    _assert_module_dict_uniqueness(module_dict)
    return nn.ModuleDict(module_dict)


def _assert_module_dict_uniqueness(module_dict: Dict[str, nn.Sequential]):
    """
    We have this function as a safeguard to help us catch if we are reusing modules
    when they should not be (i.e. if splitting into multiple branches with same layers,
    one could accidentally reuse the instantiated nn.Modules across branches).
    """
    branch_ids = [id(sequential_branch) for sequential_branch in module_dict.values()]
    assert len(branch_ids) == len(set(branch_ids))

    module_ids = []
    for sequential_branch in module_dict.values():
        module_ids += [id(module) for module in sequential_branch]

    num_total_modules = len(module_ids)
    num_unique_modules = len(set(module_ids))
    assert num_unique_modules == num_total_modules


class CNNModel(ModelBase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.conv = nn.Sequential(*_make_conv_layers(self.resblocks, self.cl_args))

        self.data_size_after_conv = pytorch_utils.calc_size_after_conv_sequence(
            self.cl_args.target_width, self.conv
        )
        self.no_out_channels = self.conv[-1].out_channels

        self.downsample_conv_identities = _get_downsample_identities_moduledict(
            num_classes=self.num_classes, in_features=self.fc_1_in_features
        )

        self.fc_1 = nn.Sequential(
            OrderedDict(
                {
                    "fc_1_bn_1": nn.BatchNorm1d(self.fc_1_in_features),
                    "fc_1_act_1": Swish(),
                    "fc_1_linear_1": nn.Linear(
                        self.fc_1_in_features, self.cl_args.fc_repr_dim, bias=False
                    ),
                }
            )
        )

        self.multi_task_branches = _get_multi_task_branches(
            num_classes=self.num_classes,
            fc_task_dim=self.fc_task_dim,
            fc_repr_and_extra_dim=self.fc_repr_and_extra_dim,
            fc_do=self.cl_args.fc_do,
        )

        self._init_weights()

    @property
    def fc_1_in_features(self) -> int:
        return self.no_out_channels * self.data_size_after_conv

    @property
    def l1_penalized_weights(self) -> torch.Tensor:
        return self.conv[0].conv_1.weight

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                # Swish slope is roughly 0.5 around 0
                nn.init.kaiming_normal_(m.weight, a=0.5, mode="fan_out")
            elif isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    @property
    def resblocks(self) -> List[int]:
        if not self.cl_args.resblocks:
            residual_blocks = find_no_resblocks_needed(
                self.cl_args.target_width,
                self.cl_args.down_stride,
                self.cl_args.first_stride_expansion,
            )
            logger.info(
                "No residual blocks specified in CL args, using input "
                "%s based on width approximation calculation.",
                residual_blocks,
            )
            return residual_blocks
        return self.cl_args.resblocks

    def forward(
        self, x: torch.Tensor, extra_inputs: torch.Tensor = None
    ) -> Dict[str, torch.Tensor]:
        out = self.conv(x)
        out = out.view(out.shape[0], -1)

        identities = _calculate_module_dict_outputs(
            input_=out, module_dict=self.downsample_conv_identities
        )

        out = self.fc_1(out)

        if extra_inputs is not None:
            out_extra = self.fc_extra(extra_inputs)
            out = torch.cat((out_extra, out), dim=1)

        out = _calculate_module_dict_outputs(
            input_=out, module_dict=self.multi_task_branches
        )

        out = {
            column_name: feature + identities[column_name]
            for column_name, feature in out.items()
        }

        return out


class MLPModel(ModelBase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.fc_0 = nn.Linear(
            self.fc_1_in_features, self.cl_args.fc_repr_dim, bias=False
        )

        self.downsample_fc_0_identities = _get_downsample_identities_moduledict(
            num_classes=self.num_classes, in_features=self.fc_repr_and_extra_dim
        )

        self.fc_1 = nn.Sequential(
            OrderedDict(
                {
                    "fc_1_bn_1": nn.BatchNorm1d(self.cl_args.fc_repr_dim),
                    "fc_1_act_1": Swish(),
                    "fc_1_linear_1": nn.Linear(
                        self.cl_args.fc_repr_dim, self.cl_args.fc_repr_dim, bias=False
                    ),
                }
            )
        )

        self.multi_task_branches = _get_multi_task_branches(
            num_classes=self.num_classes,
            fc_task_dim=self.fc_task_dim,
            fc_repr_and_extra_dim=self.fc_repr_and_extra_dim,
            fc_do=self.cl_args.fc_do,
        )

        self._init_weights()

    @property
    def fc_1_in_features(self) -> int:
        return self.cl_args.target_width * 4

    @property
    def l1_penalized_weights(self) -> torch.Tensor:
        return self.fc_0.weight

    def _init_weights(self):
        for task_branch in self.multi_task_branches.values():
            nn.init.zeros_(task_branch.fc_3_bn_1.weight)

    def forward(
        self, x: torch.Tensor, extra_inputs: torch.Tensor = None
    ) -> Dict[str, torch.Tensor]:
        out = x.view(x.shape[0], -1)

        out = self.fc_0(out)

        identity_inputs = out
        if extra_inputs is not None:
            identity_inputs = torch.cat((extra_inputs, identity_inputs), dim=1)

        identities = _calculate_module_dict_outputs(
            input_=identity_inputs, module_dict=self.downsample_fc_0_identities
        )

        out = self.fc_1(out)

        if extra_inputs is not None:
            out_extra = self.fc_extra(extra_inputs)
            out = torch.cat((out_extra, out), dim=1)

        out = _calculate_module_dict_outputs(
            input_=out, module_dict=self.multi_task_branches
        )

        out = {
            column_name: feature + identities[column_name]
            for column_name, feature in out.items()
        }

        return out


class SplitMLPModel(ModelBase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # TODO: Create constructor for MLP models
        # TODO: Account for extra inputs here

        num_chunks = self.cl_args.split_mlp_num_splits
        self.fc_0 = nn.Sequential(
            OrderedDict(
                {
                    "fc_0": SplitLinear(
                        in_features=self.fc_1_in_features,
                        out_feature_sets=self.cl_args.fc_repr_dim,
                        num_chunks=num_chunks,
                        bias=True,
                    ),
                    "fc_0_do": nn.Dropout(p=self.cl_args.fc_do),
                }
            )
        )

        in_feat = num_chunks * self.cl_args.fc_repr_dim
        self.downsample_fc_0_identities = _get_downsample_identities_moduledict(
            num_classes=self.num_classes, in_features=in_feat
        )

        self.fc_1 = nn.Sequential(
            OrderedDict(
                {
                    "fc_1_bn_1": nn.BatchNorm1d(in_feat),
                    "fc_1_act_1": Swish(),
                    "fc_1_linear_1": nn.Linear(in_feat, in_feat, bias=False),
                    "fc_1_do_1": nn.Dropout(self.cl_args.fc_do),
                }
            )
        )

        self.multi_task_branches = _get_multi_task_branches(
            num_classes=self.num_classes,
            fc_repr_and_extra_dim=in_feat,
            fc_task_dim=self.fc_task_dim,
            fc_do=self.cl_args.fc_do,
        )

        self._init_weights()

    @property
    def fc_1_in_features(self) -> int:
        return self.cl_args.target_width * 4

    @property
    def l1_penalized_weights(self) -> torch.Tensor:
        return self.fc_0[0].weight

    def _init_weights(self):
        for task_branch in self.multi_task_branches.values():
            nn.init.zeros_(task_branch.fc_3_bn_1.weight)

    def forward(
        self, x: torch.Tensor, extra_inputs: torch.Tensor = None
    ) -> Dict[str, torch.Tensor]:
        out = x.view(x.shape[0], -1)

        out = self.fc_0(out)

        identity_inputs = out
        if extra_inputs is not None:
            identity_inputs = torch.cat((extra_inputs, identity_inputs), dim=1)

        identities = _calculate_module_dict_outputs(
            input_=identity_inputs, module_dict=self.downsample_fc_0_identities
        )

        out = self.fc_1(out)

        if extra_inputs is not None:
            out_extra = self.fc_extra(extra_inputs)
            out = torch.cat((out_extra, out), dim=1)

        out = _calculate_module_dict_outputs(
            input_=out, module_dict=self.multi_task_branches
        )

        out = {
            column_name: feature + identities[column_name]
            for column_name, feature in out.items()
        }

        return out


class MGMoEModel(ModelBase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # TODO: Create constructor for MLP models

        self.num_chunks = self.cl_args.split_mlp_num_splits
        self.num_experts = self.cl_args.mg_num_experts

        fc_0_out_feat = self.num_chunks * self.cl_args.fc_repr_dim
        self.fc_0 = nn.Sequential(
            OrderedDict(
                {
                    "fc_0": SplitLinear(
                        in_features=self.fc_1_in_features,
                        out_feature_sets=self.cl_args.fc_repr_dim,
                        num_chunks=self.num_chunks,
                        bias=True,
                    ),
                    "fc_0_bn_1": nn.BatchNorm1d(fc_0_out_feat),
                    "fc_0_act_1": Swish(),
                    "fc_0_do": nn.Dropout(p=self.cl_args.fc_do),
                }
            )
        )

        self.downsample_identities = _get_downsample_identities_moduledict(
            num_classes=self.num_classes, in_features=fc_0_out_feat
        )

        gate_names = tuple(self.num_classes.keys())
        self.gates = construct_multi_branches(
            branch_names=gate_names,
            branch_layer_base=OrderedDict(
                {
                    "gate_fc": (
                        nn.Linear,
                        {
                            "in_features": fc_0_out_feat,
                            "out_features": self.num_experts,
                            "bias": True,
                        },
                    ),
                    "gate_attention": (nn.Softmax, {"dim": 1}),
                }
            ),
        )

        expert_names = tuple(f"expert_{i}" for i in range(self.num_experts))
        self.expert_branches = construct_multi_branches(
            branch_names=expert_names,
            branch_layer_base=self.get_multi_task_branch_base(
                in_features=fc_0_out_feat + self.extra_dim,
                out_features=self.fc_task_dim,
            ),
        )

        task_names = tuple(self.num_classes.keys())
        self.multi_task_branches = construct_multi_branches(
            branch_names=task_names,
            branch_layer_base=self.get_multi_task_branch_base(
                in_features=self.fc_task_dim, out_features=self.fc_task_dim
            ),
        )

        for task, num_outputs in self.num_classes.items():
            self.multi_task_branches[task].add_module(
                "final_fc", nn.Linear(self.fc_task_dim, num_outputs, bias=True)
            )

        self._init_weights()

    def get_multi_task_branch_base(
        self, in_features, out_features
    ) -> "OrderedDict[str, Tuple[nn.Module, Dict]]":
        # TODO: Move out of this class.
        # TODO: Make prefix an argument to this function, instead of hard-coding fc_1
        # TODO: Add constructor to work on the output of this function, can inject
        #       before, after and add repeats of this base by calling this with diff
        #       names

        base = OrderedDict(
            {
                "fc_1_linear_1": (
                    nn.Linear,
                    {
                        "in_features": in_features,
                        "out_features": out_features,
                        "bias": False,
                    },
                ),
                "fc_1_bn_1": (nn.BatchNorm1d, {"num_features": out_features}),
                "fc_1_act_1": (Swish, {}),
                "fc_1_do_1": (nn.Dropout, {"p": self.cl_args.fc_do}),
            }
        )

        return base

    def _init_weights(self):
        pass

    @property
    def fc_1_in_features(self) -> int:
        return self.cl_args.target_width * 4

    @property
    def l1_penalized_weights(self) -> torch.Tensor:
        return self.fc_0[0].weight

    def forward(
        self, x: torch.Tensor, extra_inputs: torch.Tensor = None
    ) -> Dict[str, torch.Tensor]:
        out = x.view(x.shape[0], -1)

        out = self.fc_0(out)

        if extra_inputs is not None:
            out = torch.cat((extra_inputs, out), dim=1)

        gate_attentions = _calculate_module_dict_outputs(
            input_=out, module_dict=self.gates
        )

        expert_outputs = _calculate_module_dict_outputs(
            input_=out, module_dict=self.expert_branches
        )

        final_out = {}
        stacked_expert_outputs = torch.stack(list(expert_outputs.values()), dim=2)
        for task_name, task_attention in gate_attentions.items():
            weighted_expert_outputs = (
                task_attention.unsqueeze(1) * stacked_expert_outputs
            )
            weighted_expert_sum = weighted_expert_outputs.sum(dim=2)

            cur_task_branch = self.multi_task_branches[task_name]
            final_out[task_name] = cur_task_branch(weighted_expert_sum)

        return final_out


def construct_multi_branches(
    branch_names: Tuple[str, ...],
    branch_layer_base: "OrderedDict[str, Tuple[nn.Module, Dict]]",
    extra_hooks: List[Callable] = (),
) -> nn.ModuleDict:
    def _initialize_module(module: nn.Module, module_args: Dict) -> nn.Module:
        return module(**module_args)

    module_dict = {}
    for name in branch_names:
        cur_branch = OrderedDict()

        for module_name, module_recipe in branch_layer_base.items():
            cur_module = module_recipe[0]
            cur_module_args = module_recipe[1]
            cur_branch[module_name] = _initialize_module(
                module=cur_module, module_args=cur_module_args
            )

        module_dict[name] = nn.Sequential(cur_branch)

    for hook in extra_hooks:
        module_dict = hook(module_dict)

    _assert_module_dict_uniqueness(module_dict)
    return nn.ModuleDict(module_dict)


def _get_downsample_identities_moduledict(
    num_classes: Dict[str, int], in_features: int
) -> nn.ModuleDict:
    """
    Currently redundant cast to `nn.Sequential` here for compatibility with
    `assert_module_dict_uniqueness`.
    """
    module_dict = {}
    for key, cur_num_output_classes in num_classes.items():
        module_dict[key] = nn.Sequential(
            nn.Linear(
                in_features=in_features, out_features=cur_num_output_classes, bias=True
            )
        )

    _assert_module_dict_uniqueness(module_dict)
    return nn.ModuleDict(module_dict)


def _calculate_module_dict_outputs(
    input_: torch.Tensor, module_dict: nn.ModuleDict
) -> Dict[str, torch.Tensor]:
    final_out = {}
    for target_column, linear_layer in module_dict.items():
        final_out[target_column] = linear_layer(input_)

    return final_out


class LinearModel(nn.Module):
    def __init__(self, cl_args: Namespace, *args, **kwargs):
        super().__init__()

        self.cl_args = cl_args
        self.fc_1_in_features = self.cl_args.target_width * 4

        self.fc_1 = nn.Linear(self.fc_1_in_features, 1)
        self.act = self._get_act()

        self.output_parser = self._get_output_parser()

    @property
    def l1_penalized_weights(self) -> torch.Tensor:
        return self.fc_1.weight

    def _get_act(self) -> Callable[[torch.Tensor], torch.Tensor]:
        if self.cl_args.target_cat_columns:
            logger.info(
                "Using logistic regression model on categorical column: %s.",
                self.cl_args.target_cat_columns,
            )
            return nn.Sigmoid()

        # no activation for linear regression
        elif self.cl_args.target_con_columns:
            logger.info(
                "Using linear regression model on continuous column: %s.",
                self.cl_args.target_con_columns,
            )
            return lambda x: x

        raise ValueError()

    def _get_output_parser(self) -> Callable[[torch.Tensor], Dict[str, torch.Tensor]]:
        def _parse_categorical(out: torch.Tensor) -> Dict[str, torch.Tensor]:
            # we create a 2D output from 1D for compatibility with visualization funcs
            out = torch.cat(((1 - out[:, 0]).unsqueeze(1), out), 1)
            out = {self.cl_args.target_cat_columns[0]: out}
            return out

        def _parse_continuous(out: torch.Tensor) -> Dict[str, torch.Tensor]:
            out = {self.cl_args.target_con_columns[0]: out}
            return out

        if self.cl_args.target_cat_columns:
            return _parse_categorical
        return _parse_continuous

    def forward(self, x: torch.Tensor, *args, **kwargs) -> Dict[str, torch.Tensor]:
        out = x.view(x.shape[0], -1)

        out = self.fc_1(out)
        out = self.act(out)

        out = self.output_parser(out)

        return out
