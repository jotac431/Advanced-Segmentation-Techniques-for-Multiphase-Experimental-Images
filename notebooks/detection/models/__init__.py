# models/__init__.py
from .maskrcnn import get_maskrcnn_model
from .deeplabv3 import get_deeplab_model

MODEL_REGISTRY = {
    "maskrcnn_resnet50_fpn": get_maskrcnn_model,
    "deeplabv3_resnet50": get_deeplab_model
}
