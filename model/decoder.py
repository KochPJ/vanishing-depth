import torch
import torch.nn as nn
import torch.nn.functional as F

from segmentation_models_pytorch.base import modules as md
try:
    from segmentation_models_pytorch.unet.decoder import DecoderBlock, CenterBlock
except:    
    from segmentation_models_pytorch.decoders.unet.decoder import UnetDecoderBlock as DecoderBlock
    from segmentation_models_pytorch.decoders.unet.decoder import UnetCenterBlock as CenterBlock


class UnetDecoder(nn.Module):
    def __init__(
            self,
            encoder_channels,
            decoder_channels,
            n_blocks=5,
            use_batchnorm=True,
            attention_type=None,
            center=False,
            fpn=False):
        super().__init__()

        if n_blocks != len(decoder_channels):
            raise ValueError(
                "Model depth is {}, but you provide `decoder_channels` for {} blocks.".format(
                    n_blocks, len(decoder_channels)
                )
            )

        encoder_channels = encoder_channels[1:]  # remove first skip with same spatial resolution
        encoder_channels = encoder_channels[::-1]  # reverse channels to start from head of encoder
        self.decoder_channels = decoder_channels

        # computing blocks input and output channels
        head_channels = encoder_channels[0]
        in_channels = [head_channels] + list(decoder_channels[:-1])
        skip_channels = list(encoder_channels[1:]) + [0]
        out_channels = decoder_channels

        if center:
            self.center = CenterBlock(
                head_channels, head_channels, use_batchnorm=use_batchnorm
            )
        else:
            self.center = nn.Identity()

        # combine decoder keyword arguments
        self.fpn = fpn

        self.version_old = True

        try:
            kwargs = dict(use_batchnorm=use_batchnorm, attention_type=attention_type)

            blocks = [
                DecoderBlock(in_ch, skip_ch, out_ch, **kwargs)
                for in_ch, skip_ch, out_ch in zip(in_channels, skip_channels, out_channels)
            ]

        except:
            kwargs = dict(use_norm=use_batchnorm, attention_type=attention_type)

            blocks = [
                DecoderBlock(in_ch, skip_ch, out_ch, **kwargs)
                for in_ch, skip_ch, out_ch in zip(in_channels, skip_channels, out_channels)
            ]
            self.version_old = False
        self.decoder_block = nn.ModuleList(blocks)


    def forward(self, features):
        
        #print(len(features))

        

        features = features[1:]    # remove first skip with same spatial resolution
        features = features[::-1]  # reverse channels to start from head of encoder

        spatial_shapes = [feature.shape[2:] for feature in features]
        
        head = features[0]
        #print('head', head.shape)
        skips = features[1:]
        #print(len(skips), [s.shape for s in skips])

        x = self.center(head)
        #print('x', x.shape)
        if self.fpn:
            outs = [x]
        for i, decoder_block in enumerate(self.decoder_block):
            skip = skips[i] if i < len(skips) else None

            #if skip is not None:
            #    print(i, x.shape, skip.shape)

            if self.version_old:                
                x = decoder_block(x, skip)
            else:
                height, width = x.shape[2:]
                x = decoder_block(x, height*2, width*2, skip)
            #print(i, x.shape)
            if self.fpn:
                outs.append(x)

        if self.fpn:
            return outs
        else:
            return x