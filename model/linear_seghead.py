import torch
import torch.nn as nn
import warnings
import torch.nn.functional as F
from segmentation_models_pytorch.base.heads import SegmentationHead


def resize(input,
           size=None,
           scale_factor=None,
           mode='nearest',
           align_corners=None,
           warning=True):
    if warning:
        if size is not None and align_corners:
            input_h, input_w = tuple(int(x) for x in input.shape[2:])
            output_h, output_w = tuple(int(x) for x in size)
            if output_h > input_h or output_w > input_w:
                if ((output_h > 1 and output_w > 1 and input_h > 1
                     and input_w > 1) and (output_h - 1) % (input_h - 1)
                        and (output_w - 1) % (input_w - 1)):
                    warnings.warn(
                        f'When align_corners={align_corners}, '
                        'the output would more aligned if '
                        f'input size {(input_h, input_w)} is `x+1` and '
                        f'out size {(output_h, output_w)} is `nx+1`')
    return F.interpolate(input, size, scale_factor, mode, align_corners)


class BNHead(nn.Module):
    """Just a batchnorm."""

    def __init__(self, in_channels=384, layers=4, patch_size=14, out_channels=1,
                 div_factor=32):
        super().__init__()
        self.in_channels = in_channels
        self.bn = nn.SyncBatchNorm(self.in_channels * layers)
        self.patch_size = patch_size
        self.out_channels = out_channels
        self.seg_head = nn.Conv2d(self.in_channels * layers, out_channels, kernel_size=1)
        self.div_factor = div_factor

        sd = torch.load('./data/backbones/dinov2_vits14_ade20k_ms_head.pth')['state_dict']
        sd = {'seg_head.weight': sd['decode_head.conv_seg.weight'],
              'seg_head.bias': sd['decode_head.conv_seg.bias'],
              'bn.weight': sd['decode_head.bn.weight'],
              'bn.bias': sd['decode_head.bn.bias'],
              'bn.running_mean': sd['decode_head.bn.running_mean'],
              'bn.running_var': sd['decode_head.bn.running_var'],
              }

        self.load_state_dict(sd)


    def forward(self, inputs, s):
        x = [x[:, 1:, :].permute(0, 2, 1).view(s[0], self.in_channels,
                                               s[2] // self.patch_size, s[3] // self.patch_size) for x in inputs]
        x = torch.cat(x, dim=1)

        #if self.div_factor is not None:
        #    out_size = ((s[2] // 32) * 32, (s[3] // 32) * 32)
        #else:
        #    out_size = (s[2], s[3])

        #x = F.interpolate(x, out_size)
        x = self.bn(x)
        x = self.seg_head(x)
        return x


if __name__ == '__main__':
    in_channels = 384

    # 644x490
    x = [torch.rand((1, in_channels, 46*35+1, 1)) for _ in range(4)]

    #resize_factors = [(490, 644) for _ in range(4)]
    m = BNHead(in_channels=in_channels) #, resize_factors=resize_factors)
    print(m)
    out = m(x, (1, 3, 490, 644))
    print(out.shape)
