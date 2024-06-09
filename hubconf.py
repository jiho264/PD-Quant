from collections import OrderedDict
from models.resnet import resnet18 as _resnet18
from models.resnet import resnet50 as _resnet50
from models.resnet import resnet152 as _resnet152
from models.mobilenetv2 import mobilenetv2 as _mobilenetv2
from models.mnasnet import mnasnet as _mnasnet
from models.regnet import regnetx_600m as _regnetx_600m
from models.regnet import regnetx_3200m as _regnetx_3200m
import torch

dependencies = ["torch"]
model_path = {
    "resnet18": "pre-trained/resnet18_imagenet.pth.tar",
    "resnet50": "pre-trained/resnet50_imagenet.pth.tar",
    "resnet152": "pre-trained/resnet152-394f9c45.pth",  # https://pytorch.org/vision/main/_modules/torchvision/models/resnet.html#ResNet152_Weights
    "mbv2": "pre-trained/mobilenetv2.pth.tar",
    "reg600m": "pre-trained/regnet_600m.pth.tar",
    "reg3200m": "pre-trained/regnet_3200m.pth.tar",
    "mnasnet": "pre-trained/mnasnet.pth.tar",
}


def resnet18(pretrained=False, **kwargs):
    # Call the model, load pretrained weights
    model = _resnet18(**kwargs)
    if pretrained:
        checkpoint = torch.load(model_path["resnet18"], map_location="cpu")
        model.load_state_dict(checkpoint)
    return model


def resnet50(pretrained=False, **kwargs):
    # Call the model, load pretrained weights
    model = _resnet50(**kwargs)
    if pretrained:
        checkpoint = torch.load(model_path["resnet50"], map_location="cpu")
        model.load_state_dict(checkpoint)
    return model


def resnet152(pretrained=False, **kwargs):
    # Call the model, load pretrained weights
    model = _resnet152(**kwargs)
    print("ResNet152_Weights_IMAGENET1K_V1")
    if pretrained:
        checkpoint = torch.load(model_path["resnet152"], map_location="cpu")
        model.load_state_dict(checkpoint)
    return model


def mobilenetv2(pretrained=False, **kwargs):
    # Call the model, load pretrained weights
    model = _mobilenetv2(**kwargs)
    if pretrained:
        checkpoint = torch.load(model_path["mbv2"], map_location="cpu")
        model.load_state_dict(checkpoint["model"])
    return model


def regnetx_600m(pretrained=False, **kwargs):
    # Call the model, load pretrained weights
    model = _regnetx_600m(**kwargs)
    if pretrained:
        checkpoint = torch.load(model_path["reg600m"], map_location="cpu")
        model.load_state_dict(checkpoint)
    return model


def regnetx_3200m(pretrained=False, **kwargs):
    # Call the model, load pretrained weights
    model = _regnetx_3200m(**kwargs)
    if pretrained:
        checkpoint = torch.load(model_path["reg3200m"], map_location="cpu")
        model.load_state_dict(checkpoint)
    return model


def mnasnet(pretrained=False, **kwargs):
    # Call the model, load pretrained weights
    model = _mnasnet(**kwargs)
    if pretrained:
        checkpoint = torch.load(model_path["mnasnet"], map_location="cpu")
        model.load_state_dict(checkpoint)
    return model
