.. _fusion-configurations:

Fusion Configurations
=====================

Configuration classes for combining multiple input modalities.

.. contents::
   :local:
   :depth: 2

Base Fusion Configuration
-------------------------

.. autoclass:: eir.setup.schemas.FusionConfig

Fusion Module Configurations
-----------------------------

.. autoclass:: eir.models.fusion.fusion_default.ResidualMLPConfig

.. autoclass:: eir.models.fusion.fusion_mgmoe.MGMoEModelConfig

.. autoclass:: eir.models.fusion.fusion_identity.IdentityConfig

.. autoclass:: eir.models.fusion.fusion_attention.AttentionFusionConfig