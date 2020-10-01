from .lenet import *
from .vgg import *
from .resnet import *
from .wideresnet import *
from .wideresnetleaky import *

"""
This package contains models for CIFAR10, CIFAR100 and MNIST datasets, every model has a grayscale parameter
to control whether it will be used for MNIST.

All models' last layer has to be named as classifier in order to finetune.
"""