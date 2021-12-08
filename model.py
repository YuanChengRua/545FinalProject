import torch, math
from torch import nn
from torch.autograd import Function
from torch.nn import functional as F

####################################################################
###################### ConvLarge Architecture ######################
####################################################################

class ConvLarge(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.block1 = self.conv_block(1, out_dim = 128)
        self.block2 = self.conv_block(128, out_dim = 128)
        self.block3 = self.conv_block(128, 128)


        self.block4 = self.conv_block(128, 256)
        self.block5 = self.conv_block(256, 256)
        self.block6 = self.conv_block(256, 128)

        self.maxpool = nn.Sequential(*[nn.MaxPool2d(kernel_size=2, stride=2), nn.Dropout2d()])
        self.fc = nn.Sequential(nn.Linear(128, num_classes))
        self.average_pool = nn.AdaptiveAvgPool2d((1, 1))

    def conv_block(self, input_dim, out_dim, kernel_size=3, stride=2, padding=2):  # 原来的是0.01 stride = 1, padding = 1, 用的是leakyrelu
        return nn.Sequential(
            nn.Conv2d(input_dim, out_dim, kernel_size, stride, padding, bias=False),
            nn.BatchNorm2d(out_dim),  # 防止 saturation
            nn.ReLU(inplace=True))
    
    def forward(self, x):
        z = self.block1(x)
        z = self.block2(z)
        z = self.block3(z)
        z = self.maxpool(z)
        z = self.block4(z)
        z = self.block5(z)
        z = self.block6(z)
        z = self.average_pool(z)
        temp = z.shape[0]
        z = z.view(temp, -1)
        z = self.fc(z)
        return z

######################################################################
###################### Shake-Shake Architecture ######################
######################################################################

def conv3x3(in_planes, out_planes, stride=1):
    "3x3 convolution with padding"
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)

class ShakeShakeBlock(nn.Module):
    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(ShakeShakeBlock, self).__init__()
        self.conv_a1 = conv3x3(inplanes, planes, stride)
        self.bn_a1 = nn.BatchNorm2d(planes)
        self.conv_a2 = conv3x3(planes, planes)
        self.bn_a2 = nn.BatchNorm2d(planes)

        self.conv_b1 = conv3x3(inplanes, planes, stride)
        self.bn_b1 = nn.BatchNorm2d(planes)
        self.conv_b2 = conv3x3(planes, planes)
        self.bn_b2 = nn.BatchNorm2d(planes)

        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        a, b, residual = x, x, x

        a = F.relu(a, inplace=False)
        a = self.conv_a1(a)
        a = self.bn_a1(a)
        a = F.relu(a, inplace=True)
        a = self.conv_a2(a)
        a = self.bn_a2(a)

        b = F.relu(b, inplace=False)
        b = self.conv_b1(b)
        b = self.bn_b1(b)
        b = F.relu(b, inplace=True)
        b = self.conv_b2(b)
        b = self.bn_b2(b)

        ab = shake(a, b, training=self.training)

        if self.downsample is not None:
            residual = self.downsample(x)

        return residual + ab

class Shake(Function):
    @staticmethod
    def forward(ctx, inp1, inp2, training):
        assert inp1.size() == inp2.size()
        gate = inp1.new(inp1.size(0), 1, 1, 1)
        if training:
            gate.uniform_(0, 1)
        else:
            gate.fill_(0.5)
        return inp1 * gate + inp2 * (1. - gate)

    @staticmethod
    def backward(ctx, grad_output):
        grad_inp1 = grad_inp2 = grad_training = None
        gate = grad_output.data.new(grad_output.size(0), 1, 1, 1).uniform_(0, 1)
        if ctx.needs_input_grad[0]:
            grad_inp1 = grad_output * gate
        if ctx.needs_input_grad[1]:
            grad_inp2 = grad_output * (1. - gate)
        assert not ctx.needs_input_grad[2]
        return grad_inp1, grad_inp2, grad_training

def shake(inp1, inp2, training=False):
    return Shake.apply(inp1, inp2, training)

class ShiftConvDownsample(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ShiftConvDownsample, self).__init__()
        self.relu = nn.ReLU(inplace=True)
        self.conv = nn.Conv2d(2*in_channels, out_channels, kernel_size=1, groups=2)
        self.bn = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        x = torch.cat((x[:, :, 0::2, 0::2], x[:, :, 1::2, 1::2]), dim=1)
        x = self.relu(x)
        x = self.conv(x)
        x = self.bn(x)
        return x

class ResNet32x32(nn.Module):
    def __init__(self, block, layers, channels, num_classes=1000, downsample='basic'):
        super(ResNet32x32, self).__init__()
        assert len(layers) == 3 and downsample in ['basic', 'shift_conv']
        self.downsample_mode = downsample
        self.inplanes = 16
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1, bias=False)
        self.layer1 = self._make_layer(block, channels, layers[0])
        self.layer2 = self._make_layer(block, channels * 2, layers[1], stride=2)
        self.layer3 = self._make_layer(block, channels * 4, layers[2], stride=2)
        self.avgpool = nn.AvgPool2d(8)
        self.fc = nn.Linear(channels * 4, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes:
            if self.downsample_mode == 'basic' or stride == 1:
                downsample = nn.Sequential(
                        nn.Conv2d(self.inplanes, planes, kernel_size=1, stride=stride, bias=False),
                        nn.BatchNorm2d(planes)
                )
            elif self.downsample_mode == 'shift_conv':
                downsample = ShiftConvDownsample(in_channels=self.inplanes, out_channels=planes)

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        return self.fc(x)

def shakeshake26(**kwargs):
    model = ResNet32x32(
            ShakeShakeBlock,
            layers=[4, 4, 4],
            channels=96,
            downsample='shift_conv', **kwargs
    )
    return model

# if __name__ == '__main__':
#     model = ConvLarge(input_dim=3)
#
#     img = torch.randn(5, 3, 32, 32)
#     logits = model(img)
#     print(logits.shape)
#
#     model = shakeshake26(num_classes=10)
#     logits = model(img)
#     print(logits.shape)