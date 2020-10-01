import torch
from torchvision import transforms
import torch.nn as nn
import numpy as np


class ZCATransformation(object):
    def __init__(self, transformation_matrix, transformation_mean):
        if transformation_matrix.size(0) != transformation_matrix.size(1):
            raise ValueError("transformation_matrix should be square. Got " +
                             "[{} x {}] rectangular matrix.".format(*transformation_matrix.size()))
        self.transformation_matrix = transformation_matrix
        self.transformation_mean = transformation_mean

    def __call__(self, tensor):
        """
        Args:
            tensor (Tensor): Tensor image of size (N, C, H, W) to be whitened.
        Returns:
            Tensor: Transformed image.
        """
        if tensor.size(0) * tensor.size(1) * tensor.size(2) != self.transformation_matrix.size(0):
            raise ValueError("tensor and transformation matrix have incompatible shape." +
                             "[{} x {} x {}] != ".format(*tensor[0].size()) +
                             "{}".format(self.transformation_matrix.size(0)))

        flat_tensor = tensor.view(1, -1)

        centered_tensor = flat_tensor - self.transformation_mean
        transformed_tensor = torch.mm(centered_tensor, self.transformation_matrix)

        tensor = transformed_tensor.view(tensor.size())
        return tensor

    def __repr__(self):
        format_string = self.__class__.__name__ + '('
        format_string += (str(self.transformation_matrix.numpy().tolist()) + ')')
        return format_string


class Rescale(object):
    def __init__(self, min, max):
        self.min = min
        self.max = max

    def __call__(self, tensor):
        return (tensor - self.min) / (self.max - self.min)


class GaussianBlur(object):
    """blur a single image on GPU"""

    def __init__(self, kernel_size):
        radias = kernel_size // 2
        kernel_size = radias * 2 + 1
        self.blur_h = nn.Conv2d(3, 3, kernel_size=(kernel_size, 1),
                                stride=1, padding=0, bias=False, groups=3)
        self.blur_v = nn.Conv2d(3, 3, kernel_size=(1, kernel_size),
                                stride=1, padding=0, bias=False, groups=3)
        self.k = kernel_size
        self.r = radias

        self.blur = nn.Sequential(
            nn.ReflectionPad2d(radias),
            self.blur_h,
            self.blur_v
        )
        self.blur = self.blur.cuda()

        self.pil_to_tensor = transforms.ToTensor()
        self.tensor_to_pil = transforms.ToPILImage()

    def __call__(self, img):
        img = self.pil_to_tensor(img).unsqueeze(0)
        img = img.cuda()

        sigma = np.random.uniform(0.1, 2.0)
        x = np.arange(-self.r, self.r + 1)
        x = np.exp(-np.power(x, 2) / (2 * sigma * sigma))
        x = x / x.sum()
        x = torch.from_numpy(x).view(1, -1).repeat(3, 1).cuda()

        self.blur_h.weight.data.copy_(x.view(3, 1, self.k, 1))
        self.blur_v.weight.data.copy_(x.view(3, 1, 1, self.k))

        with torch.no_grad():
            img = self.blur(img)
            img = img.squeeze()
            img = img.cpu()

        img = self.tensor_to_pil(img)

        return img