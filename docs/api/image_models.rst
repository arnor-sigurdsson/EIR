.. _external-image-models:

Image Models
============

This page contains the list of external image models that can be used with EIR, coming from the great `timm <https://huggingface.co/docs/timm>`__ library.

There are 3 ways to use these models:

- Configure and train specific architectures (e.g. ResNet with chosen number of layers) from scratch.
- Train a specific architecture (e.g. ``resnet18``) from scratch.
- Use a pre-trained model (e.g. ``resnet18``) and fine-tune it.

Please refer to `this page <https://huggingface.co/docs/timm/models>`__ for more detailed information about configurable architectures, and `this page <https://huggingface.co/timm>`__ for a list of pre-defined architectures, with the option of using pre-trained weights.

Configurable Models
-------------------

The following models can be configured and trained from scratch.

The model type is specified in the ``model_type`` field of the configuration, while the model specific configuration is specified in the ``model_init_config`` field.

For example, the ``ResNet`` architecture includes the ``layers`` and ``block`` parameters, and can be configured as follows:

.. literalinclude:: ../tutorials/tutorial_files/a_using_eir/05_image_tutorial/inputs.yaml
    :language: yaml
    :caption: input_configurable_image_model.yaml

.. autoclass:: timm.models.beit.Beit
   :members:
   :exclude-members: forward

.. autoclass:: timm.models.byobnet.ByobNet
   :members:
   :exclude-members: forward

.. autoclass:: timm.models.cait.Cait
   :members:
   :exclude-members: forward

.. autoclass:: timm.models.coat.CoaT
   :members:
   :exclude-members: forward

.. autoclass:: timm.models.convit.ConVit
   :members:
   :exclude-members: forward

.. autoclass:: timm.models.convmixer.ConvMixer
   :members:
   :exclude-members: forward

.. autoclass:: timm.models.convnext.ConvNeXt
   :members:
   :exclude-members: forward

.. autoclass:: timm.models.crossvit.CrossVit
   :members:
   :exclude-members: forward

.. autoclass:: timm.models.cspnet.CspNet
   :members:
   :exclude-members: forward

.. autoclass:: timm.models.davit.DaVit
   :members:
   :exclude-members: forward

.. autoclass:: timm.models.deit.VisionTransformerDistilled
   :members:
   :exclude-members: forward

.. autoclass:: timm.models.densenet.DenseNet
   :members:
   :exclude-members: forward

.. autoclass:: timm.models.dla.DLA
   :members:
   :exclude-members: forward

.. autoclass:: timm.models.dpn.DPN
   :members:
   :exclude-members: forward

.. autoclass:: timm.models.edgenext.EdgeNeXt
   :members:
   :exclude-members: forward

.. autoclass:: timm.models.efficientformer.EfficientFormer
   :members:
   :exclude-members: forward

.. autoclass:: timm.models.efficientformer_v2.EfficientFormerV2
   :members:
   :exclude-members: forward

.. autoclass:: timm.models.efficientnet.EfficientNet
   :members:
   :exclude-members: forward

.. autoclass:: timm.models.efficientvit_mit.EfficientVit
   :members:
   :exclude-members: forward

.. autoclass:: timm.models.efficientvit_msra.EfficientVitMsra
   :members:
   :exclude-members: forward

.. autoclass:: timm.models.eva.Eva
   :members:
   :exclude-members: forward

.. autoclass:: timm.models.fastvit.FastVit
   :members:
   :exclude-members: forward

.. autoclass:: timm.models.focalnet.FocalNet
   :members:
   :exclude-members: forward

.. autoclass:: timm.models.gcvit.GlobalContextVit
   :members:
   :exclude-members: forward

.. autoclass:: timm.models.ghostnet.GhostNet
   :members:
   :exclude-members: forward

.. autoclass:: timm.models.hgnet.HighPerfGpuNet
   :members:
   :exclude-members: forward

.. autoclass:: timm.models.hiera.Hiera
   :members:
   :exclude-members: forward

.. autoclass:: timm.models.hrnet.HighResolutionNet
   :members:
   :exclude-members: forward

.. autoclass:: timm.models.inception_next.MetaNeXt
   :members:
   :exclude-members: forward

.. autoclass:: timm.models.inception_resnet_v2.InceptionResnetV2
   :members:
   :exclude-members: forward

.. autoclass:: timm.models.inception_v3.InceptionV3
   :members:
   :exclude-members: forward

.. autoclass:: timm.models.inception_v4.InceptionV4
   :members:
   :exclude-members: forward

.. autoclass:: timm.models.levit.Levit
   :members:
   :exclude-members: forward

.. autoclass:: timm.models.maxxvit.MaxxVitCfg
   :members:
   :exclude-members: forward

.. autoclass:: timm.models.metaformer.MetaFormer
   :members:
   :exclude-members: forward

.. autoclass:: timm.models.mobilenetv3.MobileNetV3
   :members:
   :exclude-members: forward

.. autoclass:: timm.models.mvitv2.MultiScaleVit
   :members:
   :exclude-members: forward

.. autoclass:: timm.models.nasnet.NASNetALarge
   :members:
   :exclude-members: forward

.. autoclass:: timm.models.nest.Nest
   :members:
   :exclude-members: forward

.. autoclass:: timm.models.nextvit.NextViT
   :members:
   :exclude-members: forward

.. autoclass:: timm.models.nfnet.NormFreeNet
   :members:
   :exclude-members: forward

.. autoclass:: timm.models.pit.PoolingVisionTransformer
   :members:
   :exclude-members: forward

.. autoclass:: timm.models.pnasnet.PNASNet5Large
   :members:
   :exclude-members: forward

.. autoclass:: timm.models.pvt_v2.PyramidVisionTransformerV2
   :members:
   :exclude-members: forward

.. autoclass:: timm.models.rdnet.RDNet
   :members:
   :exclude-members: forward

.. autoclass:: timm.models.regnet.RegNet
   :members:
   :exclude-members: forward

.. autoclass:: timm.models.repghost.RepGhostNet
   :members:
   :exclude-members: forward

.. autoclass:: timm.models.repvit.RepVit
   :members:
   :exclude-members: forward

.. autoclass:: timm.models.resnet.ResNet
   :members:
   :exclude-members: forward

.. autoclass:: timm.models.resnetv2.ResNetV2
   :members:
   :exclude-members: forward

.. autoclass:: timm.models.rexnet.RexNet
   :members:
   :exclude-members: forward

.. autoclass:: timm.models.selecsls.SelecSls
   :members:
   :exclude-members: forward

.. autoclass:: timm.models.senet.SENet
   :members:
   :exclude-members: forward

.. autoclass:: timm.models.sequencer.Sequencer2d
   :members:
   :exclude-members: forward

.. autoclass:: timm.models.swin_transformer.SwinTransformer
   :members:
   :exclude-members: forward

.. autoclass:: timm.models.swin_transformer_v2.SwinTransformerV2
   :members:
   :exclude-members: forward

.. autoclass:: timm.models.swin_transformer_v2_cr.SwinTransformerV2Cr
   :members:
   :exclude-members: forward

.. autoclass:: timm.models.tiny_vit.TinyVit
   :members:
   :exclude-members: forward

.. autoclass:: timm.models.tnt.TNT
   :members:
   :exclude-members: forward

.. autoclass:: timm.models.tresnet.TResNet
   :members:
   :exclude-members: forward

.. autoclass:: timm.models.twins.Twins
   :members:
   :exclude-members: forward

.. autoclass:: timm.models.vgg.VGG
   :members:
   :exclude-members: forward

.. autoclass:: timm.models.visformer.Visformer
   :members:
   :exclude-members: forward

.. autoclass:: timm.models.vision_transformer.VisionTransformer
   :members:
   :exclude-members: forward

.. autoclass:: timm.models.vision_transformer_relpos.VisionTransformerRelPos
   :members:
   :exclude-members: forward

.. autoclass:: timm.models.vision_transformer_sam.VisionTransformerSAM
   :members:
   :exclude-members: forward

.. autoclass:: timm.models.volo.VOLO
   :members:
   :exclude-members: forward

.. autoclass:: timm.models.vovnet.VovNet
   :members:
   :exclude-members: forward

.. autoclass:: timm.models.xception.Xception
   :members:
   :exclude-members: forward

.. autoclass:: timm.models.xception_aligned.XceptionAligned
   :members:
   :exclude-members: forward

.. autoclass:: timm.models.xcit.Xcit
   :members:
   :exclude-members: forward

