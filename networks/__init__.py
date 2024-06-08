
import torch
import torch.nn as nn
from torchvision import models

from .alexnet_micro import alexnet as alexnet_m
from .resnet_micro import resnet18 as resnet18_m
from .resnet_micro import resnet34 as resnet34_m
from .densenet_micro import densenet121 as densenet121_m
from .value_model import Value_model

def load_model(name, outputsize, pretrained=None, ):

    if pretrained:
        pretrained = True
    else:
        pretrained = False
    # 不用去模型的定义处修改outputsize，直接在这里就行
    if name.lower() in 'alexnet_micro':
        model = alexnet_m(pretrained=pretrained)
        model.classifier = nn.Linear(256, outputsize)
    elif name.lower() in 'resnet18_micro':
        model = resnet18_m(pretrained=pretrained)
        model.fc = nn.Linear(model.fc.in_features, outputsize)
    elif name.lower() in 'resnet34_micro':
        model = resnet34_m(pretrained=pretrained)
        model.fc = nn.Linear(model.fc.in_features, outputsize)
    elif name.lower() in 'densenet121_micro':
        model = densenet121_m(pretrained=pretrained)
        model.classifier = nn.Linear(model.classifier.in_features, outputsize)

    return model

def load_valuemodel(input_size, hidden_size, num_classes):
    model = Value_model(input_size, hidden_size, num_classes)
    return model


