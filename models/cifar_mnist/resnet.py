import torch.nn as nn


def conv_1_3x3():
    return nn.Sequential(nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False),  # 3, 64, 7, 2, 3
                         nn.BatchNorm2d(64),
                         nn.ReLU(inplace=True))
                         # nn.MaxPool2d(kernel_size=3, stride=2, padding=1))


class bottleneck(nn.Module):
    def __init__(self, inplanes, planes, kernel_size, strides=(2, 2)):
        super(bottleneck, self).__init__()
        plane1, plane2, plane3 = planes
        self.outchannels = plane3
        self.conv1 = nn.Conv2d(inplanes, plane1, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn1 = nn.BatchNorm2d(plane1)
        self.conv2 = nn.Conv2d(plane1, plane2, kernel_size=kernel_size, stride=strides, padding=int((kernel_size - 1) / 2), bias=False)
        self.bn2 = nn.BatchNorm2d(plane2)
        self.conv3 = nn.Conv2d(plane2, plane3, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn3 = nn.BatchNorm2d(plane3)
        self.conv4 = nn.Conv2d(inplanes, plane3, kernel_size=1, stride=strides, padding=0, bias=False)
        self.bn4 = nn.BatchNorm2d(plane3)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, input_tensor):
        out = self.conv1(input_tensor)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        shortcut = self.conv4(input_tensor)
        shortcut = self.bn4(shortcut)

        out += shortcut
        out = self.relu(out)
        return out


class basic_block(nn.Module):
    def __init__(self, inplanes, outplanes, kernel_size, strides=(2, 2)):
        super(basic_block, self).__init__()

        self.conv1 = nn.Conv2d(inplanes, outplanes, kernel_size=kernel_size, stride=strides, padding=int((kernel_size - 1) / 2), bias=False)
        self.bn1 = nn.BatchNorm2d(outplanes)
        self.conv2 = nn.Conv2d(outplanes, outplanes, kernel_size=kernel_size, stride=1, padding=int((kernel_size - 1) / 2), bias=False)
        self.bn2 = nn.BatchNorm2d(outplanes)
        self.conv3 = nn.Conv2d(inplanes, outplanes, kernel_size=1, stride=strides, padding=0, bias=False)
        self.bn3 = nn.BatchNorm2d(outplanes)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, input_tensor):
        out = self.conv1(input_tensor)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        shortcut = self.conv3(input_tensor)
        shortcut = self.bn3(shortcut)

        out += shortcut
        out = self.relu(out)
        return out


class identity_block3(nn.Module):
    def __init__(self, inplanes, planes, kernel_size):
        super(identity_block3, self).__init__()
        plane1, plane2, plane3 = planes
        self.outchannels = plane3
        self.conv1 = nn.Conv2d(inplanes, plane1, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn1 = nn.BatchNorm2d(plane1)
        self.conv2 = nn.Conv2d(plane1, plane2, kernel_size=kernel_size, stride=1, padding=int((kernel_size - 1) / 2), bias=False)
        self.bn2 = nn.BatchNorm2d(plane2)
        self.conv3 = nn.Conv2d(plane2, plane3, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn3 = nn.BatchNorm2d(plane3)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, input_tensor, return_conv3_out=False):
        out = self.conv1(input_tensor)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        out += input_tensor
        out = self.relu(out)
        return out


class identity_block2(nn.Module):
    def __init__(self, inplanes, outplanes, kernel_size):
        super(identity_block2, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, outplanes, kernel_size=kernel_size, stride=1, padding=int((kernel_size - 1) / 2), bias=False)
        self.bn1 = nn.BatchNorm2d(outplanes)
        self.conv2 = nn.Conv2d(outplanes, outplanes, kernel_size=kernel_size, stride=1, padding=int((kernel_size - 1) / 2), bias=False)
        self.bn2 = nn.BatchNorm2d(outplanes)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, input_tensor, return_conv3_out=False):
        out = self.conv1(input_tensor)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        out += input_tensor
        out = self.relu(out)
        return out


class Resnet50(nn.Module):
    def __init__(self, num_classes, include_top=True):
        print('CIFAR_MNIST Resnet50 is used')
        super(Resnet50, self).__init__()
        self.num_classes = num_classes
        self.include_top = include_top
        block_ex = 4

        # Define the building blocks
        self.conv_3x3 = conv_1_3x3()

        self.bottleneck_1 = bottleneck(16 * block_ex, [16 * block_ex, 16 * block_ex, 64 * block_ex], kernel_size=3, strides=(1, 1))
        self.identity_block_1_1 = identity_block3(64*block_ex, [16*block_ex, 16*block_ex, 64*block_ex], kernel_size=3)
        self.identity_block_1_2 = identity_block3(64*block_ex, [16*block_ex, 16*block_ex, 64*block_ex], kernel_size=3)

        self.bottleneck_2 = bottleneck(64*block_ex, [32*block_ex, 32*block_ex, 128*block_ex], kernel_size=3, strides=(2, 2))
        self.identity_block_2_1 = identity_block3(128*block_ex, [32*block_ex, 32*block_ex, 128*block_ex], kernel_size=3)
        self.identity_block_2_2 = identity_block3(128*block_ex, [32*block_ex, 32*block_ex, 128*block_ex], kernel_size=3)
        self.identity_block_2_3 = identity_block3(128*block_ex, [32*block_ex, 32*block_ex, 128*block_ex], kernel_size=3)

        self.bottleneck_3 = bottleneck(128*block_ex, [64*block_ex, 64*block_ex, 256*block_ex], kernel_size=3, strides=(2, 2))
        self.identity_block_3_1 = identity_block3(256*block_ex, [64*block_ex, 64*block_ex, 256*block_ex], kernel_size=3)
        self.identity_block_3_2 = identity_block3(256*block_ex, [64*block_ex, 64*block_ex, 256*block_ex], kernel_size=3)
        self.identity_block_3_3 = identity_block3(256*block_ex, [64*block_ex, 64*block_ex, 256*block_ex], kernel_size=3)
        self.identity_block_3_4 = identity_block3(256*block_ex, [64*block_ex, 64*block_ex, 256*block_ex], kernel_size=3)
        self.identity_block_3_5 = identity_block3(256*block_ex, [64*block_ex, 64*block_ex, 256*block_ex], kernel_size=3)

        self.bottleneck_4 = bottleneck(256*block_ex, [128*block_ex, 128*block_ex, 512*block_ex], kernel_size=3, strides=(2, 2))
        self.identity_block_4_1 = identity_block3(512*block_ex, [128*block_ex, 128*block_ex, 512*block_ex], kernel_size=3)
        self.identity_block_4_2 = identity_block3(512*block_ex, [128*block_ex, 128*block_ex, 512*block_ex], kernel_size=3)

        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Linear(512*block_ex, num_classes)

        # Initialize the weights
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, input_x):
        x = self.conv_3x3(input_x)
        ret1 = x
        x = self.bottleneck_1(x)
        x = self.identity_block_1_1(x)
        x = self.identity_block_1_2(x)
        ret2 = x
        x = self.bottleneck_2(x)
        x = self.identity_block_2_1(x)
        x = self.identity_block_2_2(x)
        x = self.identity_block_2_3(x)
        ret3 = x
        x = self.bottleneck_3(x)
        x = self.identity_block_3_1(x)
        x = self.identity_block_3_2(x)
        x = self.identity_block_3_3(x)
        x = self.identity_block_3_4(x)
        x = self.identity_block_3_5(x)
        ret4 = x
        x = self.bottleneck_4(x)
        x = self.identity_block_4_1(x)
        x = self.identity_block_4_2(x)
        ret5 = x

        x = self.avgpool(x)
        if self.include_top:
            x = x.view(x.size(0), -1)
            x = self.classifier(x)
        return [ret1, ret2, ret3, ret4, ret5, x]


class Resnet18(nn.Module):
    def __init__(self, num_classes):
        print('CIFAR_MNIST Resnet18 is used')
        super(Resnet18, self).__init__()
        self.num_classes = num_classes

        # Define the building blocks
        self.conv_3x3 = conv_1_3x3()

        self.identity_block_1_0 = identity_block2(64, 64, kernel_size=3)
        self.identity_block_1_1 = identity_block2(64, 64, kernel_size=3)

        self.basic_block_2 = basic_block(64, 128, kernel_size=3, strides=(2, 2))
        self.identity_block_2_1 = identity_block2(128, 128, kernel_size=3)

        self.basic_block_3 = basic_block(128, 256, kernel_size=3, strides=(2, 2))
        self.identity_block_3_1 = identity_block2(256, 256, kernel_size=3)

        self.basic_block_4 = basic_block(256, 512, kernel_size=3, strides=(2, 2))
        self.identity_block_4_1 = identity_block2(512, 512, kernel_size=3)

        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Linear(512, num_classes)

        # Initialize the weights
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, input_x):
        x = self.conv_3x3(input_x)
        ret1 = x
        x = self.identity_block_1_0(x)
        x = self.identity_block_1_1(x)
        ret2 = x
        x = self.basic_block_2(x)
        x = self.identity_block_2_1(x)
        ret3 = x
        x = self.basic_block_3(x)
        x = self.identity_block_3_1(x)
        ret4 = x
        x = self.basic_block_4(x)
        x = self.identity_block_4_1(x)
        ret5 = x

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return [ret1, ret2, ret3, ret4, ret5, x]


def resnet18(**kwargs):
    """
    Constructs a Resnet18 model.
    """
    return Resnet18(**kwargs)


def resnet50(**kwargs):
    """
    Constructs a Resnet50 model.
    """
    return Resnet50(**kwargs)
