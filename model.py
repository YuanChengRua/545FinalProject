import torch, math
from torch import nn

from typing import Union, List, Dict, Any, cast

####################################################################
###################### ConvLarge Architecture ######################
####################################################################

class ConvLarge(nn.Module):
    def __init__(self, input_dim=3, num_classes=10, stochastic=True, top_bn=False):
        super(ConvLarge, self).__init__()
        self.block1 = self.conv_block(input_dim, 128, 3, 1, 1, 0.1)
        self.block2 = self.conv_block(128, 128, 3, 1, 1, 0.1)
        self.block3 = self.conv_block(128, 128, 3, 1, 1, 0.1)

        self.block4 = self.conv_block(128, 256, 3, 1, 1, 0.1)
        self.block5 = self.conv_block(256, 256, 3, 1, 1, 0.1)
        self.block6 = self.conv_block(256, 256, 3, 1, 1, 0.1)

        self.block7 = self.conv_block(256, 512, 3, 1, 0, 0.1)
        self.block8 = self.conv_block(512, 256, 1, 1, 0, 0.1)
        self.block9 = self.conv_block(256, 128, 1, 1, 0, 0.1)

        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))

        maxpool = [nn.MaxPool2d(kernel_size=2, stride=2)]
        if stochastic:
            maxpool.append(nn.Dropout2d())
        self.maxpool = nn.Sequential(*maxpool)

        classifier = [nn.Linear(128, num_classes)]
        if top_bn:
            classifier.append(nn.BatchNorm1d(num_classes))
        self.classifier = nn.Sequential(*classifier)

    def conv_block(self, input_dim, out_dim, kernel_size=3, stride=1, padding=1, lrelu_slope=0.01):
        return nn.Sequential(
                nn.Conv2d(input_dim, out_dim, kernel_size, stride, padding, bias=False),
                nn.BatchNorm2d(out_dim),
                nn.LeakyReLU(inplace=True, negative_slope=lrelu_slope)
        )

    def forward(self, x):
        out = self.block1(x)
        out = self.block2(out)
        out = self.block3(out)
        out = self.maxpool(out)

        out = self.block4(out)
        out = self.block5(out)
        out = self.block6(out)
        out = self.maxpool(out)

        out = self.block7(out)
        out = self.block8(out)
        out = self.block9(out)

        feature = self.avg_pool(out)
        feature = feature.view(feature.shape[0], -1)
        logits = self.classifier(feature)
        
        return logits

######################################################################
###################### Shake-Shake Architecture ######################
######################################################################


class VGG(nn.Module):

    def __init__(
        self,
        features: nn.Module,
        num_classes: int = 1000,
        init_weights: bool = True
    ) -> None:
        super(VGG, self).__init__()
        self.features = features
        self.avgpool = nn.AdaptiveAvgPool2d((7, 7))
        self.classifier = nn.Sequential(
            nn.Linear(512 * 7 * 7, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, num_classes),
        )
        if init_weights:
            self._initialize_weights()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x

    def _initialize_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)


def make_layers(cfg: List[Union[str, int]], batch_norm: bool = False) -> nn.Sequential:
    layers: List[nn.Module] = []
    in_channels = 1
    for v in cfg:
        if v == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        else:
            v = cast(int, v)
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
            else:
                layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = v
    return nn.Sequential(*layers)


cfgs: Dict[str, List[Union[str, int]]] = {
    'A': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'B': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'D': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'E': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
}
def _vgg(arch: str, cfg: str, batch_norm: bool, pretrained: bool, progress: bool, **kwargs: Any) -> VGG:

    model = VGG(make_layers(cfgs[cfg], batch_norm=batch_norm), **kwargs)

    return model
def vgg11(pretrained: bool = False, progress: bool = True, **kwargs: Any) -> VGG:
    r"""VGG 11-layer model (configuration "A") from
    `"Very Deep Convolutional Networks For Large-Scale Image Recognition" <https://arxiv.org/pdf/1409.1556.pdf>`_.
    The required minimum input size of the model is 32x32.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _vgg('vgg11', 'A', False, pretrained, progress, **kwargs)

