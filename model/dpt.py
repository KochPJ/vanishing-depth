import torch
import torch.nn as nn
import torch.nn.functional as F


def _make_scratch(in_shape, out_shape, groups=1, expand=False):
    scratch = nn.Module()

    out_shape1 = out_shape
    out_shape2 = out_shape
    out_shape3 = out_shape
    out_shape4 = out_shape
    if expand == True:
        out_shape1 = out_shape
        out_shape2 = out_shape * 2
        out_shape3 = out_shape * 4
        out_shape4 = out_shape * 8

    scratch.layer1_rn = nn.Conv2d(
        in_shape[0],
        out_shape1,
        kernel_size=3,
        stride=1,
        padding=1,
        bias=False,
        groups=groups,
    )
    scratch.layer2_rn = nn.Conv2d(
        in_shape[1],
        out_shape2,
        kernel_size=3,
        stride=1,
        padding=1,
        bias=False,
        groups=groups,
    )
    scratch.layer3_rn = nn.Conv2d(
        in_shape[2],
        out_shape3,
        kernel_size=3,
        stride=1,
        padding=1,
        bias=False,
        groups=groups,
    )
    scratch.layer4_rn = nn.Conv2d(
        in_shape[3],
        out_shape4,
        kernel_size=3,
        stride=1,
        padding=1,
        bias=False,
        groups=groups,
    )

    return scratch


def _make_resnet_backbone(resnet):
    pretrained = nn.Module()
    pretrained.layer1 = nn.Sequential(
        resnet.conv1, resnet.bn1, resnet.relu, resnet.maxpool, resnet.layer1
    )

    pretrained.layer2 = resnet.layer2
    pretrained.layer3 = resnet.layer3
    pretrained.layer4 = resnet.layer4

    return pretrained


def _make_pretrained_resnext101_wsl(use_pretrained):
    resnet = torch.hub.load("facebookresearch/WSL-Images", "resnext101_32x8d_wsl")
    return _make_resnet_backbone(resnet)


class Interpolate(nn.Module):
    """Interpolation module."""

    def __init__(self, scale_factor, mode, align_corners=False):
        """Init.

        Args:
            scale_factor (float): scaling
            mode (str): interpolation mode
        """
        super(Interpolate, self).__init__()

        self.interp = nn.functional.interpolate
        self.scale_factor = scale_factor
        self.mode = mode
        self.align_corners = align_corners

    def forward(self, x):
        """Forward pass.

        Args:
            x (tensor): input

        Returns:
            tensor: interpolated data
        """

        x = self.interp(
            x,
            scale_factor=self.scale_factor,
            mode=self.mode,
            align_corners=self.align_corners,
        )

        return x


class ResidualConvUnit(nn.Module):
    """Residual convolution module."""

    def __init__(self, features):
        """Init.

        Args:
            features (int): number of features
        """
        super().__init__()

        self.conv1 = nn.Conv2d(
            features, features, kernel_size=3, stride=1, padding=1, bias=True
        )

        self.conv2 = nn.Conv2d(
            features, features, kernel_size=3, stride=1, padding=1, bias=True
        )

        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        """Forward pass.

        Args:
            x (tensor): input

        Returns:
            tensor: output
        """
        out = self.relu(x)
        out = self.conv1(out)
        out = self.relu(out)
        out = self.conv2(out)

        return out + x


class FeatureFusionBlock(nn.Module):
    """Feature fusion block."""

    def __init__(self, features):
        """Init.

        Args:
            features (int): number of features
        """
        super(FeatureFusionBlock, self).__init__()

        self.resConfUnit1 = ResidualConvUnit(features)
        self.resConfUnit2 = ResidualConvUnit(features)

    def forward(self, *xs):
        """Forward pass.

        Returns:
            tensor: output
        """
        output = xs[0]

        if len(xs) == 2:
            output += self.resConfUnit1(xs[1])

        output = self.resConfUnit2(output)

        output = nn.functional.interpolate(
            output, scale_factor=2, mode="bilinear", align_corners=True
        )

        return output


class ResidualConvUnit_custom(nn.Module):
    """Residual convolution module."""

    def __init__(self, features, activation, bn):
        """Init.

        Args:
            features (int): number of features
        """
        super().__init__()

        self.bn = bn

        self.groups = 1

        self.conv1 = nn.Conv2d(
            features,
            features,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=not self.bn,
            groups=self.groups,
        )

        self.conv2 = nn.Conv2d(
            features,
            features,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=not self.bn,
            groups=self.groups,
        )

        if self.bn == True:
            self.bn1 = nn.BatchNorm2d(features)
            self.bn2 = nn.BatchNorm2d(features)

        self.activation = activation

        self.skip_add = nn.quantized.FloatFunctional()

    def forward(self, x):
        """Forward pass.

        Args:
            x (tensor): input

        Returns:
            tensor: output
        """

        out = self.activation(x)
        out = self.conv1(out)
        if self.bn == True:
            out = self.bn1(out)

        out = self.activation(out)
        out = self.conv2(out)
        if self.bn == True:
            out = self.bn2(out)

        if self.groups > 1:
            out = self.conv_merge(out)

        return self.skip_add.add(out, x)

        # return out + x


class FeatureFusionBlock_custom(nn.Module):
    """Feature fusion block."""

    def __init__(
        self,
        features,
        activation,
        deconv=False,
        bn=False,
        expand=False,
        align_corners=True,
    ):
        """Init.

        Args:
            features (int): number of features
        """
        super(FeatureFusionBlock_custom, self).__init__()

        self.deconv = deconv
        self.align_corners = align_corners

        self.groups = 1

        self.expand = expand
        out_features = features
        if self.expand == True:
            out_features = features // 2

        self.out_conv = nn.Conv2d(
            features,
            out_features,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=True,
            groups=1,
        )

        self.resConfUnit1 = ResidualConvUnit_custom(features, activation, bn)
        self.resConfUnit2 = ResidualConvUnit_custom(features, activation, bn)

        self.skip_add = nn.quantized.FloatFunctional()

    def forward(self, *xs):
        """Forward pass.

        Returns:
            tensor: output
        """
        output = xs[0]

        if len(xs) == 2:
            res = self.resConfUnit1(xs[1])
            output = self.skip_add.add(output, res)
            # output += res

        output = self.resConfUnit2(output)

        output = nn.functional.interpolate(
            output, scale_factor=2, mode="bilinear", align_corners=self.align_corners
        )

        output = self.out_conv(output)

        return output


def _make_fusion_block(features, use_bn):
    return FeatureFusionBlock_custom(
        features,
        nn.ReLU(False),
        deconv=False,
        bn=use_bn,
        expand=False,
        align_corners=True,
    )


class DPT(nn.Module):
    def __init__(
        self,
        features=256,
        hidden_features=128,
        use_bn=False,
        patch_size=14,
        multi_scale=False,
        num_register_tokens=0
    ):

        super(DPT, self).__init__()

        self.patch_size = patch_size
        self.in_channels = features
        self.in_features = [features//16, features // 8, features // 4, features // 2]
        self.out_channels = hidden_features
        self.multi_scale = multi_scale
        self.num_register_tokens = num_register_tokens

        # Instantiate backbone and reassemble blocks
        self.scratch = _make_scratch(in_shape=self.in_features,
                                     out_shape=self.out_channels)

        self.scratch.refinenet1 = _make_fusion_block(self.out_channels, use_bn)
        self.scratch.refinenet2 = _make_fusion_block(self.out_channels, use_bn)
        self.scratch.refinenet3 = _make_fusion_block(self.out_channels, use_bn)
        self.scratch.refinenet4 = _make_fusion_block(self.out_channels, use_bn)

        self.decoder_channels = [self.out_channels, self.out_channels, self.out_channels, self.out_channels]


        self.project4 = nn.Conv2d(
            in_channels=features,
            out_channels=self.in_features[3],
            kernel_size=1,
            stride=1,
            padding=0,
        )

        self.project3 = nn.Conv2d(
            in_channels=features,
            out_channels=self.in_features[2],
            kernel_size=1,
            stride=1,
            padding=0,
        )

        self.project2 = nn.Conv2d(
            in_channels=features,
            out_channels=self.in_features[1],
            kernel_size=1,
            stride=1,
            padding=0,
        )

        self.project1 = nn.Conv2d(
            in_channels=features,
            out_channels=self.in_features[0],
            kernel_size=1,
            stride=1,
            padding=0,
        )

    def forward(self, inputs, s=None):
        if None in inputs:
            inputs = inputs[1:]

        x = [x[:, 1+self.num_register_tokens:, :].permute(0, 2, 1)
            .view(s[0], self.in_channels, s[2] // self.patch_size, s[3] // self.patch_size)
            for x in inputs]
        #s = x[0].shape
        #x = [x_[:, :, 1:, :].view(s[0], s[1], self.i_size[0], self.i_size[1]) for x_ in x]
        #x = [F.interpolate(x_, self.o_size) for i, x_ in enumerate(x)]
        out_shape = ((s[2] // 32) * 32, (s[3] // 32) * 32)
        layer_1, layer_2, layer_3, layer_4 = x
        layer_4 = self.project4(layer_4)
        layer_3 = self.project3(layer_3)
        layer_2 = self.project2(layer_2)
        layer_1 = self.project1(layer_1)

        layer_4 = F.interpolate(layer_4, (out_shape[0] // 32, out_shape[1] // 32))
        layer_3 = F.interpolate(layer_3, (out_shape[0] // 16, out_shape[1] // 16))
        layer_2 = F.interpolate(layer_2, (out_shape[0] // 8, out_shape[1] // 8))
        layer_1 = F.interpolate(layer_1, (out_shape[0] // 4, out_shape[1] // 4))

        layer_1_rn = self.scratch.layer1_rn(layer_1)
        layer_2_rn = self.scratch.layer2_rn(layer_2)
        layer_3_rn = self.scratch.layer3_rn(layer_3)
        layer_4_rn = self.scratch.layer4_rn(layer_4)

        path_4 = self.scratch.refinenet4(layer_4_rn)
        path_3 = self.scratch.refinenet3(path_4, layer_3_rn)
        path_2 = self.scratch.refinenet2(path_3, layer_2_rn)
        path_1 = self.scratch.refinenet1(path_2, layer_1_rn)

        
        if self.multi_scale:        
            path_4 = F.interpolate(path_4, (out_shape[0] // 8, out_shape[1] // 8))
            path_3 = F.interpolate(path_3, (out_shape[0] // 4, out_shape[1] // 4))
            path_2 = F.interpolate(path_2, (out_shape[0] // 2, out_shape[1]// 2))
            path_1 = F.interpolate(path_1, (out_shape[0], out_shape[1]))
            return [path_4, path_3, path_2, path_1]
        else:
            # print('out shape 0 is in dpt.py:', out_shape[0])
            # print('out shape 1 is in dpt.py:', out_shape[1])
            layer_1 = F.interpolate(layer_1, (s[2], out_shape[1]))
            return path_1


class DPTDepthModel(DPT):
    def __init__(
        self, path=None, non_negative=True, scale=1.0, shift=0.0, invert=False, **kwargs
    ):
        features = kwargs["features"] if "features" in kwargs else 256

        self.scale = scale
        self.shift = shift
        self.invert = invert

        head = nn.Sequential(
            nn.Conv2d(features, features // 2, kernel_size=3, stride=1, padding=1),
            Interpolate(scale_factor=2, mode="bilinear", align_corners=True),
            nn.Conv2d(features // 2, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(True),
            nn.Conv2d(32, 1, kernel_size=1, stride=1, padding=0),
            nn.ReLU(True) if non_negative else nn.Identity(),
            nn.Identity(),
        )

        super().__init__(head, **kwargs)

        if path is not None:
            self.load(path)

    def forward(self, x):
        inv_depth = super().forward(x).squeeze(dim=1)

        if self.invert:
            depth = self.scale * inv_depth + self.shift
            depth[depth < 1e-8] = 1e-8
            depth = 1.0 / depth
            return depth
        else:
            return inv_depth


class DPTSegmentationModel(DPT):
    def __init__(self, num_classes, path=None, **kwargs):

        features = kwargs["features"] if "features" in kwargs else 256

        kwargs["use_bn"] = True

        head = nn.Sequential(
            nn.Conv2d(features, features, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(features),
            nn.ReLU(True),
            nn.Dropout(0.1, False),
            nn.Conv2d(features, num_classes, kernel_size=1),
            Interpolate(scale_factor=2, mode="bilinear", align_corners=True),
        )

        super().__init__(head, **kwargs)

        self.auxlayer = nn.Sequential(
            nn.Conv2d(features, features, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(features),
            nn.ReLU(True),
            nn.Dropout(0.1, False),
            nn.Conv2d(features, num_classes, kernel_size=1),
        )

        if path is not None:
            self.load(path)

class DPTSegHead(nn.Module):
    def __init__(self, num_classes, features, multi_head=False):
        super().__init__()
        self.multi_head = multi_head
        if self.multi_head:
            self.head = nn.ModuleList([
                nn.Sequential(
                    nn.Conv2d(features, features, kernel_size=3, padding=1, bias=False),
                    nn.BatchNorm2d(features),
                    nn.ReLU(True),
                    nn.Dropout(0.1, False),
                    nn.Conv2d(features, num_classes, kernel_size=1),
                    Interpolate(scale_factor=2, mode="bilinear", align_corners=True),
                )
                for _ in range(4)
            ])

        else:
            self.head = nn.Sequential(
                nn.Conv2d(features, features, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(features),
                nn.ReLU(True),
                nn.Dropout(0.1, False),
                nn.Conv2d(features, num_classes, kernel_size=1),
                Interpolate(scale_factor=2, mode="bilinear", align_corners=True),
            )

    def forward(self, x):
        if isinstance(x, list):
            if self.multi_head:
                x = [head(x_) for x_, head in zip(x, self.head)]
            else:
                x = [self.head(x_) for x_ in x]
        else:
            if self.multi_head:
                x = self.head[0](x)
            else:
                x = self.head(x)
        return x

class DPTDepthHead(nn.Module):
    def __init__(self, features, multi_head=False, activation='relu', non_negative=True, decode_factors=None):
        super().__init__()
        #print('features of DPTdepth', features)
        #input()
        self.multi_head = multi_head
        if activation == 'relu':
            act = nn.ReLU(inplace=True)
        elif activation == 'leaky-relu':
            act = nn.LeakyReLU(inplace=True)
        else:
            raise NotImplementedError('activation {} not implemented'.format(activation))
        if decode_factors is None:
            decode_factors = [1.0]

        self.decode_factors = decode_factors
        self.decode_channels = len(decode_factors)

        if isinstance(decode_factors, list):
            self.decode_dists = torch.tensor(decode_factors).view(1, self.decode_channels, 1, 1)

        if self.multi_head:
            self.head = nn.ModuleList([
                nn.Sequential(
                    nn.Conv2d(features, features // 2, kernel_size=3, stride=1, padding=1),
                    Interpolate(scale_factor=2, mode="bilinear", align_corners=True),
                    nn.Conv2d(features // 2, 32, kernel_size=3, stride=1, padding=1),
                    act,
                    nn.Conv2d(32, self.decode_channels, kernel_size=1, stride=1, padding=0),
                    act if non_negative else nn.Identity(),
                    nn.Identity(),
                )
                for _ in range(4)
            ])

        else:
            self.head = nn.Sequential(
                nn.Conv2d(features, features // 2, kernel_size=3, stride=1, padding=1),
                Interpolate(scale_factor=2, mode="bilinear", align_corners=True),
                nn.Conv2d(features // 2, 32, kernel_size=3, stride=1, padding=1),
                act,
                nn.Conv2d(32, self.decode_channels, kernel_size=1, stride=1, padding=0),
                act if non_negative else nn.Identity(),
                nn.Identity(),
        )

    def forward(self, x):
        if isinstance(x, list):
            if self.multi_head:
                x = [head(x_) for x_, head in zip(x, self.head)]
            else:
                x = [self.head(x_) for x_ in x]

            if self.decode_channels > 1:
                x = [self.decode_embs_2_dist(x_) for x_ in x]

        else:
            if self.multi_head:
                x = self.head[0](x)
            else:
                x = self.head(x)

            if self.decode_channels > 1:
                x = self.decode_embs_2_dist(x)
        if isinstance(x, list):      
            for i, tensor in enumerate(x):
                if tensor is not None: 
                    print(f'Processed in DPTDepth tensor{i} shape:{tensor.shape}')
                else:
                    print(f'Processed in DPTDepth tensor{i} is None')
        input()
        return x

    def decode_embs_2_dist(self, x):
        decode_dists = self.decode_dists.to(x.device)
        x = x.mul(decode_dists)
        x = torch.sum(x, dim=1)
        return x


if __name__ == '__main__':
    features = 384
    num_classes = 3
    # 644x490
    x = [torch.rand((1, features, 46*35 +1, 1)) for _ in range(4)]

    #resize_factors = [(490, 644) for _ in range(4)]
    m = DPT(features=features) #, resize_factors=resize_factors)
    print(m)
    out = m(x, (1, 3, 490, 644))
    print(out.shape)
    head = nn.Sequential(
        nn.Conv2d(features, features, kernel_size=3, padding=1, bias=False),
        nn.BatchNorm2d(features),
        nn.ReLU(True),
        nn.Dropout(0.1, False),
        nn.Conv2d(features, num_classes, kernel_size=1),
        Interpolate(scale_factor=2, mode="bilinear", align_corners=True),
    )
    out = head(out)
    print(out.shape)

