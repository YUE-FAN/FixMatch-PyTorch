import torch.nn as nn


class CONV_3x3(nn.Module):
    def __init__(self, inplanes, outplanes, kernelsize, stride, padding, bias):
        super(CONV_3x3, self).__init__()
        self.outchannels = outplanes
        if padding == 'same':
            p = int((kernelsize - 1) / 2)
        elif padding == 'valid':
            p = 0
        else:
            raise Exception('padding should be either same or valid')
        self.conv = nn.Conv2d(inplanes, outplanes, kernel_size=kernelsize, stride=stride, padding=p, bias=bias)
        self.bn = nn.BatchNorm2d(outplanes)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x


class VGG16(nn.Module):
    """
    Difference to original VGG16: GAP is used; BN is used

    """
    def __init__(self, grayscale, num_classes):
        super(VGG16, self).__init__()
        print("CIFAR_MNIST VGG16 is used")
        self.num_classes = num_classes
        bias = True  # by default, VGG uses bias

        # Define the building blocks
        if grayscale:
            self.conv11 = CONV_3x3(1, 64, kernelsize=3, stride=1, padding='same', bias=bias)
        else:
            self.conv11 = CONV_3x3(3, 64, kernelsize=3, stride=1, padding='same', bias=bias)
        self.conv12 = nn.Sequential(CONV_3x3(64, 64, kernelsize=3, stride=1, padding='same', bias=bias),
                                    nn.MaxPool2d(kernel_size=2, stride=2))
        self.conv21 = CONV_3x3(64, 128, kernelsize=3, stride=1, padding='same', bias=bias)
        self.conv22 = nn.Sequential(CONV_3x3(128, 128, kernelsize=3, stride=1, padding='same', bias=bias),
                                    nn.MaxPool2d(kernel_size=2, stride=2))
        self.conv31 = CONV_3x3(128, 256, kernelsize=3, stride=1, padding='same', bias=bias)
        self.conv32 = CONV_3x3(256, 256, kernelsize=3, stride=1, padding='same', bias=bias)
        self.conv33 = nn.Sequential(CONV_3x3(256, 256, kernelsize=3, stride=1, padding='same', bias=bias),
                                    nn.MaxPool2d(kernel_size=2, stride=2))
        self.conv41 = CONV_3x3(256, 512, kernelsize=3, stride=1, padding='same', bias=bias)
        self.conv42 = CONV_3x3(512, 512, kernelsize=3, stride=1, padding='same', bias=bias)
        self.conv43 = nn.Sequential(CONV_3x3(512, 512, kernelsize=3, stride=1, padding='same', bias=bias),
                                    nn.MaxPool2d(kernel_size=2, stride=2))
        self.conv51 = CONV_3x3(512, 512, kernelsize=3, stride=1, padding='same', bias=bias)
        self.conv52 = CONV_3x3(512, 512, kernelsize=3, stride=1, padding='same', bias=bias)
        self.conv53 = nn.Sequential(CONV_3x3(512, 512, kernelsize=3, stride=1, padding='same', bias=bias),
                                    nn.MaxPool2d(kernel_size=2, stride=2))

        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Linear(512, 4096)
        self.bn1 = nn.BatchNorm1d(4096)
        self.relu1 = nn.ReLU(True)
        self.fc2 = nn.Linear(4096, 4096)
        self.bn2 = nn.BatchNorm1d(4096)
        self.relu2 = nn.ReLU(True)
        self.classifier = nn.Linear(4096, num_classes)

        # Initialize the weights
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, input_x):
        x = self.conv11(input_x)
        x = self.conv12(x)
        ret1 = x
        x = self.conv21(x)
        x = self.conv22(x)
        ret2 = x
        x = self.conv31(x)
        x = self.conv32(x)
        x = self.conv33(x)
        ret3 = x
        x = self.conv41(x)
        x = self.conv42(x)
        x = self.conv43(x)
        ret4 = x
        x = self.conv51(x)
        x = self.conv52(x)
        x = self.conv53(x)
        ret5 = x
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        x = self.bn1(x)
        x = self.relu1(x)
        x = self.fc2(x)
        x = self.bn2(x)
        x = self.relu2(x)
        x = self.classifier(x)
        return [ret1, ret2, ret3, ret4, ret5, x]


def vgg16(**kwargs):
    """
    Constructs a VGG16 model.
    """
    return VGG16(**kwargs)