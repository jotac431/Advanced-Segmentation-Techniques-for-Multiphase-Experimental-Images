# models/__init__.py
from .maskrcnn import get_maskrcnn_resnet50_fpn, get_maskrcnn_resnet50_fpn_v2, get_maskrcnn_resnet101_fpn, get_maskrcnn_mobilenet
from .deeplabv3 import get_deeplab_model
from .unet import get_unet

MODEL_REGISTRY = {
    "maskrcnn_resnet50_fpn": get_maskrcnn_resnet50_fpn,
    "maskrcnn_resnet50_fpn_v2": get_maskrcnn_resnet50_fpn_v2,
    "maskrcnn_resnet101_fpn": get_maskrcnn_resnet101_fpn,
    "maskrcnn_mobilenet": get_maskrcnn_mobilenet,
    "deeplabv3_resnet50": get_deeplab_model,
    "unet": get_unet
}
