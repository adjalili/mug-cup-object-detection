import torch
import torchvision
from scipy.cluster.hierarchy import weighted
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

def build_model(backbone: str):
    if backbone == "fasterrcnn_resnet50_fpn":

        weight = torchvision.models.detection.FasterRCNN_ResNet50_FPN_Weights.DEFAULT
        model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True, weights=weight)
    else:

        weight = torchvision.models.detection.FasterRCNN_MobileNet_V3_Large_FPN_Weights.DEFAULT
        model = torchvision.models.detection.fasterrcnn_mobilenet_v3_large_fpn(pretrained=True, weights=weight)


    return model