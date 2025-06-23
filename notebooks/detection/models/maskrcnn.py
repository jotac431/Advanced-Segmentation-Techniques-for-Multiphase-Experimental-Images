#models/maskrcnn.py
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor
from torchvision.models.detection import MaskRCNN
from torchvision.models.detection.backbone_utils import resnet_fpn_backbone

from torchvision.models.detection.backbone_utils import _validate_trainable_layers
from torchvision.ops.feature_pyramid_network import LastLevelMaxPool
from torchvision.models import mobilenet_v3_large


def get_maskrcnn_resnet50_fpn(num_classes):
    # load an instance segmentation model pre-trained on COCO
    model = torchvision.models.detection.maskrcnn_resnet50_fpn(weights="DEFAULT")
    return _replace_heads(model, num_classes)

def get_maskrcnn_resnet50_fpn_v2(num_classes):
    # load an instance segmentation model pre-trained on COCO
    model = torchvision.models.detection.maskrcnn_resnet50_fpn_v2(weights="DEFAULT")
    return _replace_heads(model, num_classes)

def get_maskrcnn_resnet101_fpn(num_classes):
    backbone = resnet_fpn_backbone('resnet101', weights="DEFAULT")
    model = MaskRCNN(backbone=backbone, num_classes=num_classes)
    return model

from torchvision.models.detection.backbone_utils import BackboneWithFPN
from torchvision.models import mobilenet_v3_large
from collections import OrderedDict
import torch
from torchvision.models.detection.rpn import AnchorGenerator

def get_maskrcnn_mobilenet(num_classes):
    # 1️⃣ Load mobilenet backbone
    backbone_model = mobilenet_v3_large(weights="IMAGENET1K_V1").features

    # 2️⃣ Define which layers to extract
    return_layers = {
        '4': '0',   # low
        '11': '1',  # mid
        '13': '2'   # high
    }
    in_channels_list = [40, 112, 160]
    out_channels = 256

    # 3️⃣ Build the FPN
    backbone = BackboneWithFPN(
        backbone_model,
        return_layers=return_layers,
        in_channels_list=in_channels_list,
        out_channels=out_channels
    )


    # ❗ Match anchor generator with the number of FPN outputs (3 in this case)
    anchor_generator = AnchorGenerator(
        sizes=((32,), (64,), (128,), (256,)),  # 4 levels = 4 tuples
        aspect_ratios=((0.5, 1.0, 2.0),) * 4   # same aspect ratios for each
    )

    # 4️⃣ Create the model with the custom backbone and anchor generator
    model = MaskRCNN(
        backbone,
        num_classes=num_classes,
        rpn_anchor_generator=anchor_generator
    )

    return model



def _replace_heads(model, num_classes):

    # get number of input features for the classifier
    in_features = model.roi_heads.box_predictor.cls_score.in_features

    # replace the pre-trained head with a new one
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    # now get the number of input features for the mask classifier
    in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
    hidden_layer = 256

    # and replace the mask predictor with a new one
    model.roi_heads.mask_predictor = MaskRCNNPredictor(
        in_features_mask, hidden_layer, num_classes
    )

    return model
