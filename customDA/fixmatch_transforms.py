from torchvision import transforms
from customDA.randaugment import RandAugmentMC
from customDA import np_transforms
import numpy as np
import torch


class TransformFix_CIFAR(object):
    def __init__(self, mean, std, num_weak=1, num_strong=1):
        self.num_weak = num_weak
        self.num_strong = num_strong
        self.weak = transforms.Compose([
            transforms.ToPILImage(),
            transforms.RandomHorizontalFlip(),  # with p = 0.5
            transforms.RandomCrop(32, padding=4, padding_mode='reflect'),  # with p = 1
            np_transforms.PILToNumpy(),  # PIL Image -> np.uint8
            np_transforms.NumpyToTensor(),  # np.uint8 -> torch.float32
            np_transforms.Normalize_11()
            ])
        self.strong = transforms.Compose([
            transforms.ToPILImage(),
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(size=32,
                                  padding=int(32*0.125),
                                  padding_mode='reflect'),
            RandAugmentMC(n=2, m=10),
            np_transforms.PILToNumpy(),  # PIL Image -> np.uint8
            np_transforms.NumpyToTensor(),  # np.uint8 -> torch.float32
            np_transforms.Normalize_11(),
            ])

    def __call__(self, x):
        returnlist = []
        for i in range(self.num_weak):
            returnlist.append(self.weak(x))
        for i in range(self.num_strong):
            returnlist.append(self.strong(x))
        return returnlist
