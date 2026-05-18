"""
models/sensor_encoders.py
=========================
공유 센서 인코더 컴포넌트 re-export 모듈.

모든 멀티모달 모델(LSTM, LWM, LWM_Temporal, Chiron)이
동일한 인코더 구현을 공유하기 위한 단일 진입점.

Classes
-------
ImageTokenEncoder     : ResNet18 frozen backbone → (B, 49, D)
PointNetEncoder       : PointNet-style attention pool → (B, 16, D)
GatedCrossModalFusion : Gated cross-attention fusion block
"""

from models.fusion_blocks import GatedCrossModalFusion
from models.image_encoders import ImageTokenEncoder
from models.lidar_encoders import PointNetEncoder

__all__ = [
    "ImageTokenEncoder",
    "PointNetEncoder",
    "GatedCrossModalFusion",
]
