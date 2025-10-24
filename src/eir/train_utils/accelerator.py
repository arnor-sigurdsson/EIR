import torch
from lightning import Fabric

from eir.setup.config import Configs
from eir.utils.logging import get_logger

logger = get_logger(name=__name__)


def setup_accelerator(configs: Configs) -> Fabric:
    gc = configs.global_config

    original_hardware = gc.accelerator.hardware

    if torch.cuda.is_available() and (original_hardware in ["cuda", "gpu", "auto"]):
        # Workaround for Lightning Fabric Incompatibility
        # PyTorch >= 2.9 deprecated legacy precision flags, but Lightning Fabric
        # still uses the legacy getter (torch.get_float32_matmul_precision),
        # causing a RuntimeError if the new API is used.
        # We must use the old API until Fabric is updated.
        # https://github.com/Lightning-AI/pytorch-lightning/issues/21256

        # 1. Legacy API for matmul (cuBLAS)
        torch.set_float32_matmul_precision("high")

        # 2. Legacy API for convolutions (cuDNN)
        torch.backends.cudnn.allow_tf32 = True

        logger.info(
            "Set matmul precision to 'medium' and enabled cuDNN TF32 (Legacy API)"
        )

        # Intended Future Code (post-Lightning update)
        # torch.backends.cuda.matmul.fp32_precision = "tf32"
        # type: ignore[attr-defined]
        # torch.backends.cudnn.fp32_precision = "tf32"
        # type: ignore[attr-defined]
        # logger.info(
        #     "Set float32 matmul/cuDNN precision to 'tf32'
        #     for Tensor Core optimization"
        # )

    original_precision = gc.accelerator.precision
    original_strategy = gc.accelerator.strategy
    original_devices = gc.accelerator.devices

    fabric = Fabric(
        accelerator=original_hardware,
        precision=original_precision,
        strategy=original_strategy,
        devices=original_devices,
        num_nodes=gc.accelerator.num_nodes,
    )

    actual_accelerator = fabric.accelerator.__class__.__name__
    actual_strategy = fabric.strategy.__class__.__name__
    actual_devices = fabric.world_size if hasattr(fabric, "world_size") else 1
    actual_precision = original_precision

    logger.info(
        f"Using accelerator: {actual_accelerator}"
        + (
            f" (auto-determined from '{original_hardware}')"
            if original_hardware == "auto"
            else ""
        )
    )

    logger.info(
        f"Using strategy: {actual_strategy}"
        + (
            f" (auto-determined from '{original_strategy}')"
            if original_strategy == "auto"
            else ""
        )
    )

    logger.info(
        f"Using precision: {actual_precision}"
        + (
            f" (auto-determined from '{original_precision}')"
            if original_precision == "auto"
            else ""
        )
    )

    logger.info(
        f"Using devices: {actual_devices}"
        + (
            f" (auto-determined from '{original_devices}')"
            if original_devices == "auto"
            else ""
        )
    )

    num_nodes = fabric.node_rank + 1 if hasattr(fabric, "node_rank") else 1
    logger.info(f"Using num_nodes: {num_nodes}")

    needs_launch = False

    if fabric.world_size > 1:
        needs_launch = True

    distributed_strategies = [
        "DataParallelStrategy",
        "DDPStrategy",
        "DeepSpeedStrategy",
        "FSDPStrategy",
    ]
    if any(strat in actual_strategy for strat in distributed_strategies):
        needs_launch = True

    if getattr(fabric, "_launched", False):
        needs_launch = False

    if needs_launch:
        logger.info("Launching distributed environment")
        fabric.launch()
    else:
        logger.info("Running in single-process mode")

    return fabric
