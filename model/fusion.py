from torch import nn
import torch
import torch.nn.functional as F


class ConvDepthFusion(nn.Module):
    def __init__(self, channels: int, num_fusion_layers: int = 3):
        super().__init__()
        if num_fusion_layers > 1:
            c = channels * 2
            step = int(channels / num_fusion_layers)
            self.fusion = []
            for i in range(num_fusion_layers-1):
                self.fusion.append(nn.Conv2d(c, c-step, kernel_size=1))
                c -= step

            self.fusion.append(nn.Conv2d(c, channels, kernel_size=1))
            self.fusion = nn.Sequential(*self.fusion)
        else:
            self.fusion = nn.Conv2d(channels*2, channels, kernel_size=1)

    def forward(self, x, depth_x):
        return self.fusion(torch.cat((x, depth_x), dim=1))


class SqueezeAndExcitation(nn.Module):
    '''
    Copied from https://github.com/TUI-NICR/ESANet
    paper title: Efficient RGB-D Semantic Segmentation for Indoor Scene Analysis
    authros: Seichter, Daniel and K{\"o}hler, Mona and Lewandowski, Benjamin and Wengefeld, Tim and Gross, Horst-Michael
    '''
    def __init__(self, channel, reduction=16, activation=nn.ReLU(inplace=True)):
        super(SqueezeAndExcitation, self).__init__()
        self.fc = nn.Sequential(
            nn.Conv2d(channel, channel // reduction, kernel_size=1),
            activation,
            nn.Conv2d(channel // reduction, channel, kernel_size=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        weighting = F.adaptive_avg_pool2d(x, 1)
        weighting = self.fc(weighting)
        y = x * weighting
        return y


class SqueezeAndExciteFusionAdd(nn.Module):
    '''
        Copied from https://github.com/TUI-NICR/ESANet
        paper title: Efficient RGB-D Semantic Segmentation for Indoor Scene Analysis
        authros: Seichter, Daniel and K{\"o}hler, Mona and Lewandowski, Benjamin and Wengefeld, Tim and Gross, Horst-Michael
    '''
    def __init__(self, channels, activation=nn.ReLU(inplace=True), return_rgb_depth_p=True, eval_only_rdps=False):
        super(SqueezeAndExciteFusionAdd, self).__init__()

        self.se_rgb = SqueezeAndExcitation(channels, activation=activation)
        self.se_depth = SqueezeAndExcitation(channels, activation=activation)
        self.return_rgb_depth_p = return_rgb_depth_p
        self.eval_only_rdps = eval_only_rdps

    def forward(self, rgb, depth):
        rgb = self.se_rgb(rgb)
        depth = self.se_depth(depth)
        out = rgb + depth
        if self.return_rgb_depth_p:
            if self.eval_only_rdps and not self.training:
                return out
            s = torch.stack((torch.abs(rgb), torch.abs(depth))).detach()
            s = s.div(s.sum(0) + 1e-6).flatten(1).mean(1)
            s = s.div(torch.sum(s))
            return (out, s[-1])
        return out