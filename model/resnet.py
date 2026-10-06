
import torch.nn as nn
import torch
import torchvision.models as models
import model.fusion as dFusion
import model.crossmodalfusion as cmFusion
import inspect
from torchvision.models.feature_extraction import create_feature_extractor

__resnets__ = {
    '18': models.resnet18,
    '34': models.resnet34,
    '50': models.resnet50,
    '101': models.resnet101,
    '152': models.resnet152,
}


__resnets_weights__ = {
    '18': models.ResNet18_Weights,
    '34': models.ResNet34_Weights,
    '50': models.ResNet50_Weights,
    '101': models.ResNet101_Weights,
    '152': models.ResNet152_Weights,
}


__resnets_channels__ = {
    '18': (64, 64, 128, 256, 512),
    '34': (64, 64, 128, 256, 512),
    '50': (64, 256, 512, 1024, 2048),
    '101': (64, 256, 512, 1024, 2048),
    '152': (64, 256, 512, 1024, 2048),
}

__fusion__ = {
    'Squeeze&Excite': dFusion.SqueezeAndExciteFusionAdd,
    'Conv': dFusion.ConvDepthFusion,
    'FeatureRectifyModule': cmFusion.FeatureRectifyModule,
    'FeatureFusionModule': cmFusion.FeatureFusionModule
}


class ResNet(nn.Module):
    def __init__(self, encoder: str, num_classes: int, pretrained: bool = False, return_nodes=None):
        super(ResNet, self).__init__()
        if pretrained:
            weights = __resnets_weights__
        else:
            weights = None

        self.arch = __resnets__[encoder](weights=__resnets_weights__[encoder].DEFAULT)
        self.channels = __resnets_channels__[encoder]
        if num_classes > 0:
            self.arch.fc = nn.Linear(self.arch.fc.in_features, num_classes, bias=True)
            self.out_channels = num_classes
        else:
            self.out_channels = self.arch.fc.in_features
            self.arch.fc = nn.Identity()

        self.interm_channels = [256, 512, 1024, 2048]

        self.return_nodes = return_nodes
        self.patch_size = 16
        print('channels', self.channels)

        if return_nodes is not None:
            self.arch = create_feature_extractor(self.arch, return_nodes=self.return_nodes)
        
    def forward(self, x):
        out = self.arch(x)
        if isinstance(out, dict):
            out = list(out.values())
            out.append(None)

        return out
        


class ResNetRGBD(nn.Module):
    def __init__(self, encoder: str, num_classes: int, pretrained: bool = False, fusion='Squeeze&Excite',
                 fuse_layers: bool = True, return_layers: bool = False, with_fc: bool = True,
                 num_fusion_layers: int = 3, depth_channels: int = 1, with_depth: bool = True,
                 return_as_list: bool = True, multi_head_classification=False, depth_classification: int = 0):
        super(ResNetRGBD, self).__init__()

        self.num_classes = num_classes
        self.encoder = encoder
        self.return_layers = return_layers
        self.return_as_list = return_as_list
        self.fuse_layers = fuse_layers
        self.with_fc = with_fc if num_classes > 0 else False
        c = num_classes if with_fc else 0
        self.color_encoder = ResNet(encoder, c, pretrained=pretrained)
        self.with_depth = with_depth
        self.out_channels = self.color_encoder.out_channels
        self.fusion_layer0 = None
        self.fusion_layer1 = None
        self.fusion_layer2 = None
        self.fusion_layer3 = None
        self.fusion_layer4 = None
        self.multi_head_classification = multi_head_classification
        self.depth_classification = depth_classification
        if self.multi_head_classification:
            self.color_fc = nn.Linear(self.color_encoder.in_features, self.out_channels)
        else:
            self.color_fc = None

        if with_depth:
            self.depth_encoder = ResNet(encoder, 0, pretrained=False)
            if depth_channels != 3:
                self.depth_encoder.arch.conv1 = nn.Conv2d(depth_channels,
                                                          64,
                                                          kernel_size=tuple(self.depth_encoder.arch.conv1.kernel_size),
                                                          stride=tuple(self.depth_encoder.arch.conv1.stride),
                                                          padding=tuple(self.depth_encoder.arch.conv1.padding),
                                                          bias=bool(self.depth_encoder.arch.conv1.bias))
                if depth_classification > 0:
                    self.depth_encoder.arch.fc = nn.Linear(self.depth_encoder.in_features, depth_classification)
                else:
                    self.depth_encoder.arch.fc = nn.Identity()

            self.fuse = True if fusion is not None else False
            self.channels = __resnets_channels__[encoder]
            fusion = __fusion__[fusion]
            args = [arg.name for arg in inspect.signature(fusion).parameters.values()]
            arg_dict = {'channels': self.channels[0], 'num_fusion_layers': num_fusion_layers}
            if self.fuse and fuse_layers:
                self.fusion_layer0 = fusion(**{arg: arg_dict[arg] for arg in args if arg in arg_dict})
                arg_dict['channels'] = self.channels[1]
                self.fusion_layer1 = fusion(**{arg: arg_dict[arg] for arg in args if arg in arg_dict})
                arg_dict['channels'] = self.channels[2]
                self.fusion_layer2 = fusion(**{arg: arg_dict[arg] for arg in args if arg in arg_dict})
                arg_dict['channels'] = self.channels[3]
                self.fusion_layer3 = fusion(**{arg: arg_dict[arg] for arg in args if arg in arg_dict})
            arg_dict['channels'] = self.channels[4]
            self.fusion_layer4 = fusion(**{arg: arg_dict[arg] for arg in args if arg in arg_dict})

    def forward(self, x, xd=None):

        xs = {}
        # init layers + resnet first layer
        if self.return_layers:
            xs['input'] = x
        x = self.color_encoder.arch.conv1(x)
        x = self.color_encoder.arch.bn1(x)
        x = self.color_encoder.arch.relu(x)

        if xd is not None and self.with_depth:
            xd = self.depth_encoder.arch.conv1(xd)
            xd = self.depth_encoder.arch.bn1(xd)
            xd = self.depth_encoder.arch.relu(xd)

        if self.fuse_layers and xd is not None and self.with_depth:
            x = self.fusion_layer0(x, xd)

        if self.return_layers:
            xs['0'] = x

        # first layer
        x = self.color_encoder.arch.maxpool(x)
        x = self.color_encoder.arch.layer1(x)

        if xd is not None and self.with_depth:
            xd = self.depth_encoder.arch.maxpool(xd)
            xd = self.depth_encoder.arch.layer1(xd)

        if self.fuse_layers and xd is not None and self.with_depth:
            x = self.fusion_layer1(x, xd)

        if self.return_layers:
            xs['1'] = x

        # second layer
        x = self.color_encoder.arch.layer2(x)

        if xd is not None and self.with_depth:
            xd = self.depth_encoder.arch.layer2(xd)

            if self.fuse_layers:
                x = self.fusion_layer2(x, xd)

        if self.return_layers:
            xs['2'] = x

        # third layer
        x = self.color_encoder.arch.layer3(x)

        if xd is not None and self.with_depth:
            xd = self.depth_encoder.arch.layer3(xd)

            if self.fuse_layers:
                x = self.fusion_layer3(x, xd)

        if self.return_layers:
            xs['3'] = x

        # fourth layer
        x = self.color_encoder.arch.layer4(x)

        if self.multi_head_classification:
            xs['x'] = self.color_encoder.arch.avgpool(x)
            xs['x'] = torch.flatten(xs['x'], 1)
            xs['x'] = self.color_fc(xs['x'])

        if xd is not None and self.with_depth:
            xd = self.depth_encoder.arch.layer4(xd)
            if self.multi_head_classification:
                xs['xd'] = self.depth_encoder.arch.avgpool(x)
                xs['xd'] = torch.flatten(xs['xd'], 1)
                xs['xd'] = self.depth_encoder.arch.fc(xs['xd'])
            x = self.fusion_layer4(x, xd)


        if self.return_layers:
            xs['4'] = x

        if self.with_fc:
            x = self.color_encoder.arch.avgpool(x)
            x = torch.flatten(x, 1)
            x = self.color_encoder.arch.fc(x)
            xs['out'] = x

        if self.multi_head_classification:
            x = {'out': xs['out'], 'x': xs['x'], 'xd': xs.get('xd')}

        if self.return_layers:
            if self.return_as_list:
                return list(xs.values())
            else:
                return xs
        else:
            return x


class ResNetv2RGBD(nn.Module):
    def __init__(self, encoder: str, num_classes: int, pretrained: bool = False, fusion='Squeeze&Excite',
                 fuse_layers: bool = True, return_layers: bool = False, with_fc: bool = True,
                 num_fusion_layers: int = 3, depth_channels: int = 1, with_depth: bool = True,
                 return_as_list: bool = True, return_rdps: bool = False, multi_head_classification=False,
                 depth_classification: int = 0):

        super(ResNetv2RGBD, self).__init__()
        self.num_classes = num_classes
        self.encoder = encoder
        self.return_rdps = return_rdps
        self.return_layers = return_layers
        self.fuse_layers = fuse_layers
        self.with_fc = with_fc if num_classes > 0 else False
        c = num_classes if with_fc else 0
        self.color_encoder = ResNet(encoder, c, pretrained=pretrained)
        self.with_depth = with_depth
        self.return_as_list = return_as_list
        self.out_channels = self.color_encoder.out_channels
        self.fusion_layer0 = None
        self.fusion_layer1 = None
        self.fusion_layer2 = None
        self.fusion_layer3 = None
        self.fusion_layer4 = None
        self.multi_head_classification = multi_head_classification
        self.depth_classification = depth_classification
        if self.multi_head_classification:
            self.color_fc = nn.Linear(self.color_encoder.fc.in_features,  self.out_channels)
        else:
            self.color_fc = None

        if with_depth:
            self.depth_encoder = ResNet(encoder, 0, pretrained=False)
            if depth_channels != 3:
                self.depth_encoder.arch.conv1 = nn.Conv2d(depth_channels,
                                                          64,
                                                          kernel_size=tuple(self.depth_encoder.arch.conv1.kernel_size),
                                                          stride=tuple(self.depth_encoder.arch.conv1.stride),
                                                          padding=tuple(self.depth_encoder.arch.conv1.padding),
                                                          bias=bool(self.depth_encoder.arch.conv1.bias))
                if depth_classification > 0:
                    self.depth_encoder.arch.fc = nn.Linear(self.out_channels,
                                                           self.depth_classification)
                    self.out_channels = self.depth_classification
                else:
                    self.depth_encoder.arch.fc = nn.Identity()

            self.fuse = True if fusion is not None else False
            self.channels = __resnets_channels__[encoder]
            fusion = __fusion__[fusion]
            args = [arg.name for arg in inspect.signature(fusion).parameters.values()]
            arg_dict = {'channels': self.channels[0], 'num_fusion_layers': num_fusion_layers}
            if self.fuse and fuse_layers:
                self.fusion_layer0 = fusion(**{arg: arg_dict[arg] for arg in args if arg in arg_dict})
                arg_dict['channels'] = self.channels[1]
                self.fusion_layer1 = fusion(**{arg: arg_dict[arg] for arg in args if arg in arg_dict})
                arg_dict['channels'] = self.channels[2]
                self.fusion_layer2 = fusion(**{arg: arg_dict[arg] for arg in args if arg in arg_dict})
                arg_dict['channels'] = self.channels[3]
                self.fusion_layer3 = fusion(**{arg: arg_dict[arg] for arg in args if arg in arg_dict})
            arg_dict['channels'] = self.channels[4]
            self.fusion_layer4 = fusion(**{arg: arg_dict[arg] for arg in args if arg in arg_dict})

    def forward(self, x, xd=None):

        xs = {}
        rdps = {}
        # init layers + resnet first layer
        if self.return_layers:
            xs['input'] = x
        x = self.color_encoder.arch.conv1(x)
        x = self.color_encoder.arch.bn1(x)
        x = self.color_encoder.arch.relu(x)

        if xd is not None and self.with_depth:
            xd = self.depth_encoder.arch.conv1(xd)
            xd = self.depth_encoder.arch.bn1(xd)
            xd = self.depth_encoder.arch.relu(xd)

        if self.fuse_layers and xd is not None and self.with_depth:
            xd = self.fusion_layer0(x, xd)
            if isinstance(xd, tuple):
                xd, rdp = xd
                rdps['0'] = rdp

        if self.return_layers:
            xs['0'] = xd

        # first layer
        x = self.color_encoder.arch.maxpool(x)
        x = self.color_encoder.arch.layer1(x)

        if xd is not None and self.with_depth:
            xd = self.depth_encoder.arch.maxpool(xd)
            xd = self.depth_encoder.arch.layer1(xd)

        if self.fuse_layers and xd is not None and self.with_depth:
            xd = self.fusion_layer1(x, xd)
            if isinstance(xd, tuple):
                xd, rdp = xd
                rdps['1'] = rdp

        if self.return_layers:
            xs['1'] = xd

        # second layer
        x = self.color_encoder.arch.layer2(x)

        if xd is not None and self.with_depth:
            xd = self.depth_encoder.arch.layer2(xd)

            if self.fuse_layers:
                xd = self.fusion_layer2(x, xd)
                if isinstance(xd, tuple):
                    xd, rdp = xd
                    rdps['2'] = rdp

        if self.return_layers:
            xs['2'] = xd

        # third layer
        x = self.color_encoder.arch.layer3(x)

        if xd is not None and self.with_depth:
            xd = self.depth_encoder.arch.layer3(xd)

            if self.fuse_layers:
                xd = self.fusion_layer3(x, xd)
                if isinstance(xd, tuple):
                    xd, rdp = xd
                    rdps['3'] = rdp

        if self.return_layers:
            xs['3'] = xd

        # fourth layer
        x = self.color_encoder.arch.layer4(x)

        if self.multi_head_classification:
            xs['x'] = self.color_encoder.arch.avgpool(x)
            xs['x'] = torch.flatten(xs['x'], 1)
            xs['x'] = self.color_fc(xs['x'])

        if xd is not None and self.with_depth:
            xd = self.depth_encoder.arch.layer4(xd)
            if self.multi_head_classification:
                xs['xd'] = self.depth_encoder.arch.avgpool(x)
                xs['xd'] = torch.flatten(xs['xd'], 1)
                xs['xd'] = self.depth_encoder.arch.fc(xs['xd'])

            if self.depth_classification > 0:
                xd_cls = self.color_encoder.arch.avgpool(xd)
                xd_cls = torch.flatten(xd_cls, 1)
                xd_cls = self.color_encoder.arch.fc(xd_cls)
                xs['out_d'] = xd_cls

            xd = self.fusion_layer4(x, xd)
            if isinstance(xd, tuple):
                xd, rdp = xd
                rdps['4'] = rdp

        if self.return_layers:
            xs['4'] = xd

        if self.with_fc:
            x = self.color_encoder.arch.avgpool(xd)
            x = torch.flatten(x, 1)
            x = self.color_encoder.arch.fc(x)
            xs['out'] = x

        if self.multi_head_classification:
            x = {'out': xs['out'], 'x': xs['x'], 'xd': xs.get('xd')}

        if self.return_layers:
            if self.return_as_list:
                if rdps and self.return_rdps:
                    return (list(xs.values()), rdps)
                else:
                    return list(xs.values())
            else:
                if rdps and self.return_rdps:
                    return (xs, rdps)
                else:
                    return xs
        else:
            if rdps and self.return_rdps:
                return (x, rdps)
            else:
                return x


class ResNetv3RGBD(nn.Module):
    def __init__(self, encoder: str, num_classes: int, pretrained: bool = False,
                 fuse_layers: bool = True, return_layers: bool = False, with_fc: bool = True,
                 num_fusion_layers: int = 3, depth_channels: int = 1, with_depth: bool = True,
                 return_as_list: bool = True):
        super(ResNetv3RGBD, self).__init__()

        self.num_classes = num_classes
        self.encoder = encoder
        self.return_layers = return_layers
        self.fuse_layers = fuse_layers
        self.with_fc = with_fc if num_classes > 0 else False
        c = num_classes if with_fc else 0
        self.color_encoder = ResNet(encoder, c, pretrained=pretrained)
        self.with_depth = with_depth
        self.return_as_list = return_as_list
        self.out_channels = self.color_encoder.out_channels
        self.fusion_layer0 = None
        self.fusion_layer1 = None
        self.fusion_layer2 = None
        self.fusion_layer3 = None
        self.fusion_layer4 = None

        if with_depth:
            self.depth_encoder = ResNet(encoder, 0, pretrained=False)
            if depth_channels != 3:
                self.depth_encoder.arch.conv1 = nn.Conv2d(depth_channels,
                                                          64,
                                                          kernel_size=tuple(self.depth_encoder.arch.conv1.kernel_size),
                                                          stride=tuple(self.depth_encoder.arch.conv1.stride),
                                                          padding=tuple(self.depth_encoder.arch.conv1.padding),
                                                          bias=bool(self.depth_encoder.arch.conv1.bias))
                self.depth_encoder.arch.fc = nn.Identity()

            self.fuse = True
            self.channels = __resnets_channels__[encoder]
            fusion = __fusion__['FeatureRectifyModule']
            if self.return_layers:
                feature_fusion = __fusion__['FeatureFusionModule']
            else:
                feature_fusion = None
            args = [arg.name for arg in inspect.signature(fusion).parameters.values()]
            arg_dict = {'channels': self.channels[0], 'num_fusion_layers': num_fusion_layers}
            if self.fuse and fuse_layers:
                self.fusion_layer0 = fusion(**{arg: arg_dict[arg] for arg in args if arg in arg_dict})
                if self.return_layers:
                    self.feature_fusion_layer0 = feature_fusion(**{arg: arg_dict[arg] for arg in args if arg in arg_dict})
                else:
                    self.feature_fusion_layer0 = None
                arg_dict['channels'] = self.channels[1]
                self.fusion_layer1 = fusion(**{arg: arg_dict[arg] for arg in args if arg in arg_dict})
                if self.return_layers:
                    self.feature_fusion_layer1 = feature_fusion(**{arg: arg_dict[arg] for arg in args if arg in arg_dict})
                else:
                    self.feature_fusion_layer1 = None
                arg_dict['channels'] = self.channels[2]
                self.fusion_layer2 = fusion(**{arg: arg_dict[arg] for arg in args if arg in arg_dict})
                if self.return_layers:
                    self.feature_fusion_layer2 = feature_fusion(**{arg: arg_dict[arg] for arg in args if arg in arg_dict})
                else:
                    self.feature_fusion_layer2 = None
                arg_dict['channels'] = self.channels[3]
                self.fusion_layer3 = fusion(**{arg: arg_dict[arg] for arg in args if arg in arg_dict})
                if self.return_layers:
                    self.feature_fusion_layer3 = feature_fusion(**{arg: arg_dict[arg] for arg in args if arg in arg_dict})
                else:
                    self.feature_fusion_layer3 = None
            arg_dict['channels'] = self.channels[4]
            self.fusion_layer4 = fusion(**{arg: arg_dict[arg] for arg in args if arg in arg_dict})
            self.feature_fusion_layer4 = feature_fusion(**{arg: arg_dict[arg] for arg in args if arg in arg_dict})

    def forward(self, x, xd=None):

        xs = {}
        # init layers + resnet first layer
        if self.return_layers:
            xs['input'] = x
        x = self.color_encoder.arch.conv1(x)
        x = self.color_encoder.arch.bn1(x)
        x = self.color_encoder.arch.relu(x)

        if xd is not None and self.with_depth:
            xd = self.depth_encoder.arch.conv1(xd)
            xd = self.depth_encoder.arch.bn1(xd)
            xd = self.depth_encoder.arch.relu(xd)

        if self.fuse_layers and xd is not None and self.with_depth:
            x, xd = self.fusion_layer0(x, xd)

        if self.return_layers:
            xs['0'] = self.feature_fusion_layer0(x, xd)

        # first layer
        x = self.color_encoder.arch.maxpool(x)
        x = self.color_encoder.arch.layer1(x)

        if xd is not None and self.with_depth:
            xd = self.depth_encoder.arch.maxpool(xd)
            xd = self.depth_encoder.arch.layer1(xd)

        if self.fuse_layers and xd is not None and self.with_depth:
            x, xd = self.fusion_layer1(x, xd)

        if self.return_layers:
            xs['1'] = self.feature_fusion_layer1(x, xd)

        # second layer
        x = self.color_encoder.arch.layer2(x)

        if xd is not None and self.with_depth:
            xd = self.depth_encoder.arch.layer2(xd)

            if self.fuse_layers:
                x, xd = self.fusion_layer2(x, xd)

        if self.return_layers:
            xs['2'] = self.feature_fusion_layer2(x, xd)

        # third layer
        x = self.color_encoder.arch.layer3(x)

        if xd is not None and self.with_depth:
            xd = self.depth_encoder.arch.layer3(xd)

            if self.fuse_layers:
                x, xd = self.fusion_layer3(x, xd)

        if self.return_layers:
            xs['3'] = self.feature_fusion_layer3(x, xd)

        # fourth layer
        x = self.color_encoder.arch.layer4(x)

        if xd is not None and self.with_depth:
            xd = self.depth_encoder.arch.layer4(xd)
            x, xd = self.fusion_layer4(x, xd)

        xd = self.feature_fusion_layer4(x, xd)
        if self.return_layers:
            xs['4'] = xd

        if self.with_fc:
            x = self.color_encoder.arch.avgpool(xd)
            x = torch.flatten(x, 1)
            x = self.color_encoder.arch.fc(x)
            xs['out'] = x

        if self.return_layers:
            if self.return_as_list:
                return list(xs.values())
            else:
                return xs
        else:
            return x



class ResNetv4RGBD(nn.Module):
    def __init__(self, encoder: str, num_classes: int, pretrained: bool = False, fusion='Squeeze&Excite',
                 fuse_layers: bool = True, return_layers: bool = False, with_fc: bool = True,
                 num_fusion_layers: int = 3, depth_channels: int = 1, with_depth: bool = True,
                 return_as_list: bool = True, return_rdps: bool = False, multi_head_classification=False):
        super(ResNetv4RGBD, self).__init__()

        self.num_classes = num_classes
        self.encoder = encoder
        self.return_rdps = return_rdps
        self.return_layers = return_layers
        self.fuse_layers = fuse_layers
        self.with_fc = with_fc if num_classes > 0 else False
        c = num_classes if with_fc else 0
        self.color_encoder = ResNet(encoder, c, pretrained=pretrained)
        self.with_depth = with_depth
        self.return_as_list = return_as_list
        self.out_channels = self.color_encoder.out_channels
        self.fusion_layer0 = None
        self.fusion_layer1 = None
        self.fusion_layer2 = None
        self.fusion_layer3 = None
        self.fusion_layer4 = None
        self.multi_head_classification = multi_head_classification
        if self.multi_head_classification:
            self.color_fc = nn.Linear(self.color_encoder.fc.in_features, self.out_channels)
        else:
            self.color_fc = None

        if with_depth:
            self.depth_encoder = ResNet(encoder, 0, pretrained=False)
            if depth_channels != 3:
                self.depth_encoder.arch.conv1 = nn.Conv2d(depth_channels,
                                                          64,
                                                          kernel_size=tuple(self.depth_encoder.arch.conv1.kernel_size),
                                                          stride=tuple(self.depth_encoder.arch.conv1.stride),
                                                          padding=tuple(self.depth_encoder.arch.conv1.padding),
                                                          bias=bool(self.depth_encoder.arch.conv1.bias))
                self.depth_encoder.arch.fc = nn.Identity()

            self.fuse = True if fusion is not None else False
            self.channels = __resnets_channels__[encoder]
            fusion = __fusion__[fusion]
            args = [arg.name for arg in inspect.signature(fusion).parameters.values()]
            k = 2
            if self.return_layers:
                k+= 1
            arg_dict = {'channels': self.channels[0], 'num_fusion_layers': num_fusion_layers}
            if self.fuse and fuse_layers:
                self.fusion_layer0 = [fusion(**{arg: arg_dict[arg] for arg in args if arg in arg_dict})
                                      for _ in range(k)]
                arg_dict['channels'] = self.channels[1]
                self.fusion_layer1 = [fusion(**{arg: arg_dict[arg] for arg in args if arg in arg_dict})
                                      for _ in range(k)]
                arg_dict['channels'] = self.channels[2]
                self.fusion_layer2 = [fusion(**{arg: arg_dict[arg] for arg in args if arg in arg_dict})
                                      for _ in range(k)]
                arg_dict['channels'] = self.channels[3]
                self.fusion_layer3 = [fusion(**{arg: arg_dict[arg] for arg in args if arg in arg_dict})
                                      for _ in range(k)]
            arg_dict['channels'] = self.channels[4]
            self.fusion_layer4 = fusion(**{arg: arg_dict[arg] for arg in args if arg in arg_dict})

    def forward(self, x, xd=None):

        xs = {}
        rdps = {}
        # init layers + resnet first layer
        if self.return_layers:
            xs['input'] = x
        x = self.color_encoder.arch.conv1(x)
        x = self.color_encoder.arch.bn1(x)
        x = self.color_encoder.arch.relu(x)

        if xd is not None and self.with_depth:
            xd = self.depth_encoder.arch.conv1(xd)
            xd = self.depth_encoder.arch.bn1(xd)
            xd = self.depth_encoder.arch.relu(xd)

        if self.return_layers:
            xs['0'] = self.fusion_layer0[2](x, xd)

        if self.fuse_layers and xd is not None and self.with_depth:
            xd = self.fusion_layer0[0](x, xd)
            if isinstance(xd, tuple):
                xd, rdp = xd
                rdps['0'] = rdp
            x = self.fusion_layer0[1](x, xd)

        # first layer
        x = self.color_encoder.arch.maxpool(x)
        x = self.color_encoder.arch.layer1(x)

        if xd is not None and self.with_depth:
            xd = self.depth_encoder.arch.maxpool(xd)
            xd = self.depth_encoder.arch.layer1(xd)

        if self.return_layers:
            xs['1'] = self.fusion_layer1[2](x, xd)

        if self.fuse_layers and xd is not None and self.with_depth:
            xd = self.fusion_layer1[0](x, xd)
            if isinstance(xd, tuple):
                xd, rdp = xd
                rdps['1'] = rdp
            x = self.fusion_layer1[1](x, xd)

        # second layer
        x = self.color_encoder.arch.layer2(x)

        if self.return_layers:
            xs['2'] = self.fusion_layer2[2](x, xd)

        if xd is not None and self.with_depth:
            xd = self.depth_encoder.arch.layer2(xd)

            if self.fuse_layers:
                xd = self.fusion_layer2[0](x, xd)
                if isinstance(xd, tuple):
                    xd, rdp = xd
                    rdps['2'] = rdp
                x = self.fusion_layer2[1](x, xd)

        # third layer
        x = self.color_encoder.arch.layer3(x)

        if self.return_layers:
            xs['3'] = self.fusion_layer3[2](x, xd)

        if xd is not None and self.with_depth:
            xd = self.depth_encoder.arch.layer3(xd)

            if self.fuse_layers:
                xd = self.fusion_layer3[0](x, xd)
                if isinstance(xd, tuple):
                    xd, rdp = xd
                    rdps['3'] = rdp
                x = self.fusion_layer3[1](x, xd)

        # fourth layer
        x = self.color_encoder.arch.layer4(x)

        if self.multi_head_classification:
            xs['x'] = self.color_encoder.arch.avgpool(x)
            xs['x'] = torch.flatten(xs['x'], 1)
            xs['x'] = self.color_fc(xs['x'])

        if xd is not None and self.with_depth:
            xd = self.depth_encoder.arch.layer4(xd)

            if self.multi_head_classification:
                xs['xd'] = self.depth_encoder.arch.avgpool(x)
                xs['xd'] = torch.flatten(xs['xd'], 1)
                xs['xd'] = self.depth_encoder.arch.fc(xs['xd'])

            xd = self.fusion_layer4(x, xd)
            if isinstance(xd, tuple):
                xd, rdp = xd
                rdps['4'] = rdp

        if self.return_layers:
            xs['4'] = xd

        if self.with_fc:
            x = self.color_encoder.arch.avgpool(xd)
            x = torch.flatten(x, 1)
            x = self.color_encoder.arch.fc(x)
            xs['out'] = x

        if self.multi_head_classification:
            x = {'out': xs['out'], 'x': xs['x'], 'xd': xs.get('xd')}

        if self.return_layers:
            if self.return_as_list:
                if rdps and self.return_rdps:
                    return (list(xs.values()), rdps)
                else:
                    return list(xs.values())
            else:
                if rdps and self.return_rdps:
                    return (xs, rdps)
                else:
                    return xs
        else:
            if rdps and self.return_rdps:
                return (x, rdps)
            else:
                return x



if __name__ == '__main__':
    with torch.no_grad():
        for name, r in __resnets__.items():
            print('name: ', name)
            m = ResNet(name, 10)
            m.eval()
            print(m(torch.rand(1, 3, 224, 224)).shape)
            for f in __fusion__.keys():
                print('fusion: ', f)
                m = ResNetRGBD(name, 10, fusion=f)
                m.eval()
                print(m(torch.rand(1, 3, 224, 224), torch.rand(1, 1, 224, 224)).shape)
                print('fusion without layers: ', f)
                m = ResNetRGBD(name, 10, fusion=f, fuse_layers=False)
                m.eval()
                print(m(torch.rand(1, 3, 224, 224), torch.rand(1, 1, 224, 224)).shape)