"""
models
======
Channel prediction model zoo for the CARLA-Wireless dataset.

Models:
    ChironChannelPredictor   -- channel-only, factorized spatio-temporal
    ChironMultiModalPredictor -- multi-modal (channel + image + ego state)
    MultimodalChannelPredictor -- legacy baseline
    MultiModalPredictator      -- 2nd-gen token-level fusion baseline
    MSCPMultiModalPredictor    -- MSCP-inspired channel + sensing/scene model
"""

from models.chiron_channel import ChironChannelPredictor

__all__ = [
    "ChironChannelPredictor",
]

# Other models are available but require additional dependencies (torchvision etc.)
# Uncomment as needed:
# from models.components import ImageEncoder, ChannelEncoder, CrossAttentionFusion, PredictionHead
# from models.channel_predictor import MultimodalChannelPredictor
# from models.multi_modal_predictator import MultiModalPredictator
# from models.mscp_multimodal import MSCPMultiModalPredictor, MSCPSceneEncoder
# from models.chiron_multimodal import ChironMultiModalPredictor
