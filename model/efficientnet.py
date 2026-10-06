
import torch.nn as nn
import torch
import torchvision.models as models
import model.fusion as dFusion
import inspect

__efficientNets__ = {
    'B0': models.efficientnet_b0,
    'B1': models.efficientnet_b1,
    'B2': models.efficientnet_b2,
    'B3': models.efficientnet_b3,
    'B4': models.efficientnet_b4,
    'B5': models.efficientnet_b5,
    'B6': models.efficientnet_b6,
    'B7': models.efficientnet_b7
}

__efficient_channels__ = {
    'B0': (16, 24, 40, 112, 1280),
    'B1': (16, 24, 40, 112, 1280),
    'B2': (16, 24, 48, 120, 1408),
    'B3': (24, 32, 48, 136, 1536),
    'B4': (24, 32, 56, 160, 1792),
    'B5': (24, 40, 64, 176, 2048),
    'B6': (32, 40, 72, 200, 2304),
    'B7': (32, 48, 80, 224, 2560)
}

__fusion__ = {
    'Squeeze&Excite': dFusion.SqueezeAndExciteFusionAdd,
    'Conv': dFusion.ConvDepthFusion,
}


class EfficientNet(nn.Module):
    def __init__(self, encoder: str, num_classes: int, pretrained: bool = False):
        super(EfficientNet, self).__init__()
        self.arch = __efficientNets__[encoder](pretrained=pretrained)
        self.channels = __efficient_channels__[encoder]
        if num_classes > 0:
            self.arch.classifier[1] = nn.Linear(self.arch.classifier[1].in_features, num_classes, bias=True)
            self.out_channels = num_classes
        else:
            self.out_channels = self.arch.classifier[1].in_features
            self.arch.classifier = nn.Identity()

    def forward(self, x):
        return self.arch(x)


class EfficientNetRGBD(nn.Module):
    def __init__(self, encoder: str, num_classes: int, pretrained: bool = False, fusion='Squeeze&Excite',
                 fuse_layers: bool = True, return_layers: bool = False, with_fc: bool = True,
                 num_fusion_layers: int = 3, depth_channels: int = 1, with_depth: bool = True,
                 return_as_list: bool = True):
        super(EfficientNetRGBD, self).__init__()

        self.num_classes = num_classes
        self.encoder = encoder
        self.return_layers = return_layers
        self.return_as_list = return_as_list
        self.fuse_layers = fuse_layers
        self.with_fc = with_fc if num_classes > 0 else False
        c = num_classes if with_fc else 0
        self.color_encoder = EfficientNet(encoder, c, pretrained=pretrained)
        self.with_depth = with_depth
        self.out_channels = self.color_encoder.out_channels
        self.fusion_layer0 = None
        self.fusion_layer1 = None
        self.fusion_layer2 = None
        self.fusion_layer3 = None
        self.fusion_layer4 = None

        if with_depth:
            self.depth_encoder = EfficientNet(encoder, 0, pretrained=False)
            if depth_channels != 3:
                self.depth_encoder.arch.features[0][0] = nn.Conv2d(depth_channels,
                                                          self.depth_encoder.arch.features[0][0].out_channels,
                                                          kernel_size=tuple(self.depth_encoder.arch.features[0][0].kernel_size),
                                                          stride=tuple(self.depth_encoder.arch.features[0][0].stride),
                                                          padding=tuple(self.depth_encoder.arch.features[0][0].padding),
                                                          bias=bool(self.depth_encoder.arch.features[0][0].bias))
                self.depth_encoder.arch.classifier = nn.Identity()

            self.fuse = True if fusion is not None else False
            self.channels = __efficient_channels__[encoder]
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

    def forward(self, x, xd):
        xs = {}
        # init layers + resnet first layer
        if self.return_layers:
            xs['input'] = x

        # first layer
        x = self.color_encoder.arch.features[0](x)
        x = self.color_encoder.arch.features[1](x)

        if xd is not None and self.with_depth:
            xd = self.depth_encoder.arch.features[0](xd)
            xd = self.depth_encoder.arch.features[1](xd)

            if self.fuse_layers:
                x = self.fusion_layer0(x, xd)

        if self.return_layers:
            xs['0'] = x

        # second layer
        x = self.color_encoder.arch.features[2](x)

        if xd is not None and self.with_depth:
            xd = self.depth_encoder.arch.features[2](xd)

            if self.fuse_layers:
                x = self.fusion_layer1(x, xd)

        if self.return_layers:
            xs['1'] = x

        # third layer
        x = self.color_encoder.arch.features[3](x)

        if xd is not None and self.with_depth:
            xd = self.depth_encoder.arch.features[3](xd)

            if self.fuse_layers:
                x = self.fusion_layer2(x, xd)

        if self.return_layers:
            xs['2'] = x

        # fourth layer
        x = self.color_encoder.arch.features[4](x)
        x = self.color_encoder.arch.features[5](x)
        if xd is not None and self.with_depth:
            xd = self.depth_encoder.arch.features[4](xd)
            xd = self.depth_encoder.arch.features[5](xd)
            if self.fuse_layers:
                x = self.fusion_layer3(x, xd)

        if self.return_layers:
            xs['3'] = x

        # fith layer
        x = self.color_encoder.arch.features[6](x)
        x = self.color_encoder.arch.features[7](x)
        x = self.color_encoder.arch.features[8](x)
        if xd is not None and self.with_depth:
            xd = self.depth_encoder.arch.features[6](xd)
            xd = self.depth_encoder.arch.features[7](xd)
            xd = self.depth_encoder.arch.features[8](xd)
            x = self.fusion_layer4(x, xd)

        if self.return_layers:
            xs['4'] = x

        if self.with_fc:
            x = self.color_encoder.arch.avgpool(x)
            x = torch.flatten(x, 1)
            x = self.color_encoder.arch.classifier(x)
            if self.return_layers:
                xs['out'] = x

        if self.return_layers:
            if self.return_as_list:
                return list(xs.values())
            else:
                return xs
        else:
            return x


class EfficientNetRGBDv2(nn.Module):
    def __init__(self, encoder: str, num_classes: int, pretrained: bool = False, fusion='Squeeze&Excite',
                 fuse_layers: bool = True, return_layers: bool = False, with_fc: bool = True,
                 num_fusion_layers: int = 3, depth_channels: int = 1, with_depth: bool = True,
                 return_as_list: bool = True):
        super(EfficientNetRGBDv2, self).__init__()

        self.num_classes = num_classes
        self.encoder = encoder
        self.return_layers = return_layers
        self.return_as_list = return_as_list
        self.fuse_layers = fuse_layers
        self.with_fc = with_fc if num_classes > 0 else False
        c = num_classes if with_fc else 0
        self.color_encoder = EfficientNet(encoder, c, pretrained=pretrained)
        self.with_depth = with_depth
        self.out_channels = self.color_encoder.out_channels
        self.fusion_layer0 = None
        self.fusion_layer1 = None
        self.fusion_layer2 = None
        self.fusion_layer3 = None
        self.fusion_layer4 = None

        if with_depth:
            self.depth_encoder = EfficientNet(encoder, 0, pretrained=False)
            if depth_channels != 3:
                self.depth_encoder.arch.features[0][0] = nn.Conv2d(depth_channels,
                                                          self.depth_encoder.arch.features[0][0].out_channels,
                                                          kernel_size=tuple(self.depth_encoder.arch.features[0][0].kernel_size),
                                                          stride=tuple(self.depth_encoder.arch.features[0][0].stride),
                                                          padding=tuple(self.depth_encoder.arch.features[0][0].padding),
                                                          bias=bool(self.depth_encoder.arch.features[0][0].bias))
                self.depth_encoder.arch.classifier = nn.Identity()

            self.fuse = True if fusion is not None else False
            channels = __efficient_channels__[encoder]
            fusion = __fusion__[fusion]
            args = [arg.name for arg in inspect.signature(fusion).parameters.values()]
            arg_dict = {'channels': channels[0], 'num_fusion_layers': num_fusion_layers}
            if self.fuse and fuse_layers:
                self.fusion_layer0 = fusion(**{arg: arg_dict[arg] for arg in args if arg in arg_dict})
                arg_dict['channels'] = channels[1]
                self.fusion_layer1 = fusion(**{arg: arg_dict[arg] for arg in args if arg in arg_dict})
                arg_dict['channels'] = channels[2]
                self.fusion_layer2 = fusion(**{arg: arg_dict[arg] for arg in args if arg in arg_dict})
                arg_dict['channels'] = channels[3]
                self.fusion_layer3 = fusion(**{arg: arg_dict[arg] for arg in args if arg in arg_dict})
            arg_dict['channels'] = channels[4]
            self.fusion_layer4 = fusion(**{arg: arg_dict[arg] for arg in args if arg in arg_dict})

    def forward(self, x, xd):

        #print('input', x.shape, depth.shape)
        xs = {}
        # init layers + resnet first layer
        if self.return_layers:
            xs['input'] = x

        # first layer
        x = self.color_encoder.arch.features[0](x)
        x = self.color_encoder.arch.features[1](x)

        if xd is not None and self.with_depth:
            xd = self.depth_encoder.arch.features[0](xd)
            xd = self.depth_encoder.arch.features[1](xd)

            if self.fuse_layers:
                xd = self.fusion_layer0(x, xd)

        if self.return_layers:
            xs['0'] = xd

        # second layer
        x = self.color_encoder.arch.features[2](x)

        if xd is not None and self.with_depth:
            xd = self.depth_encoder.arch.features[2](xd)

            if self.fuse_layers:
                xd = self.fusion_layer1(x, xd)

        if self.return_layers:
            xs['1'] = xd

        # third layer
        x = self.color_encoder.arch.features[3](x)

        if xd is not None and self.with_depth:
            xd = self.depth_encoder.arch.features[3](xd)

            if self.fuse_layers:
                xd = self.fusion_layer2(x, xd)

        if self.return_layers:
            xs['2'] = xd

        # fourth layer
        x = self.color_encoder.arch.features[4](x)
        x = self.color_encoder.arch.features[5](x)
        if xd is not None and self.with_depth:
            xd = self.depth_encoder.arch.features[4](xd)
            xd = self.depth_encoder.arch.features[5](xd)
            if self.fuse_layers:
                xd = self.fusion_layer3(x, xd)

        if self.return_layers:
            xs['3'] = xd

        # fith layer
        x = self.color_encoder.arch.features[6](x)
        x = self.color_encoder.arch.features[7](x)
        x = self.color_encoder.arch.features[8](x)
        if xd is not None and self.with_depth:
            xd = self.depth_encoder.arch.features[6](xd)
            xd = self.depth_encoder.arch.features[7](xd)
            xd = self.depth_encoder.arch.features[8](xd)
            xd = self.fusion_layer4(x, xd)

        if self.return_layers:
            xs['4'] = xd

        if self.with_fc:
            x = self.color_encoder.arch.avgpool(xd)
            x = torch.flatten(x, 1)
            x = self.color_encoder.arch.classifier(x)
            if self.return_layers:
                xs['out'] = x

        if self.return_layers:
            if self.return_as_list:
                return list(xs.values())
            else:
                return xs
        else:
            return x


if __name__ == '__main__':
    with torch.no_grad():
        for name in ['B3', 'B5']:
            print('name: ', name)
            r = __efficientNets__[name]
            m = EfficientNet(name, 10, pretrained=True)
            print(m.channels)
            m.eval()

            print(m(torch.rand(1, 3, 224, 224)).shape)
            for f in __fusion__.keys():
                print('fusion: ', f)
                m = EfficientNetRGBDv2(name, 10, fusion=f, depth_channels=64)
                m.eval()

                try:
                    print(m(torch.rand(1, 3, 224, 224), torch.rand(1, 64, 224, 224)).shape)
                except Exception as e:
                    import copy
                    x = torch.rand(1, 3, 224, 224)
                    print('test', x.shape)
                    for i, f in enumerate(m.color_encoder.arch.features):
                        x = f(x)
                        print('test', i, x.shape)
                    raise e

                print('fusion without layers: ', f)
                m = EfficientNetRGBDv2(name, 10, fusion=f, fuse_layers=False)
                m.eval()
                print(m(torch.rand(1, 3, 224, 224), torch.rand(1, 1, 224, 224)).shape)