'''
Copied and edidted from:
https://github.com/facebookresearch/dino/blob/7c446df5b9f45747937fb0d72314eb9f7b66930a/vision_transformer.py#L257

'''
import os
import sys
import os.path

import torch.nn as nn
import torch
import warnings
import math
from math import lcm
import numpy as np
import torch.nn.functional as F
from functools import partial
from model.fusion import SqueezeAndExciteFusionAdd
from model.dinov2.models.vision_transformer import vit_small as dinov2_small
from model.dinov2.models.vision_transformer import vit_base as dinov2_base
from model.dinov2.models.vision_transformer import vit_large as dinov2_large
from model.dinov3.models.vision_transformer import vit_small as dinov3_small
from model.dinov3.models.vision_transformer import vit_base as dinov3_base
from model.dinov3.models.vision_transformer import vit_large as dinov3_large

from utils.stuff import load_fitting_state_dict
import transformers
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def drop_path(x, drop_prob: float = 0., training: bool = False):
    if drop_prob == 0. or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)  # work with diff dim tensors, not just 2D ConvNets
    random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
    random_tensor.floor_()  # binarize
    output = x.div(keep_prob) * random_tensor
    return output


class DropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample  (when applied in main path of residual blocks).
    """
    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x, attn


class Block(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, x, return_attention=False):
        y, attn = self.attn(self.norm1(x))
        if return_attention:
            return attn
        x = x + self.drop_path(y)
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


class PatchEmbed(nn.Module):
    """ Image to Patch Embedding
    """
    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=768):
        super().__init__()
        num_patches = (img_size // patch_size) * (img_size // patch_size)
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = num_patches

        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        #B, C, H, W = x.shape
        x = self.proj(x).flatten(2).transpose(1, 2)
        return x


class VisionTransformer(nn.Module):
    """ Vision Transformer """
    def __init__(self, img_size=[224], patch_size=16, in_chans=3, num_classes=0, embed_dim=768, depth=12,
                 num_heads=12, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop_rate=0., attn_drop_rate=0.,
                 drop_path_rate=0., norm_layer=nn.LayerNorm, return_layers=None, return_rdps=False, add_fusion=False,
                 **kwargs):
        super().__init__()
        self.num_features = self.embed_dim = embed_dim

        self.patch_embed = PatchEmbed(
            img_size=img_size[0], patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim)
        num_patches = self.patch_embed.num_patches

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))
        self.pos_drop = nn.Dropout(p=drop_rate)

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  # stochastic depth decay rule
        self.blocks = nn.ModuleList([
            Block(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer)
            for i in range(depth)])
        self.norm = norm_layer(embed_dim)

        # Classifier head
        self.head = nn.Linear(embed_dim, num_classes) if num_classes > 0 else nn.Identity()

        trunc_normal_(self.pos_embed, std=.02)
        trunc_normal_(self.cls_token, std=.02)
        self.apply(self._init_weights)

        self.fusion_layers = None
        self.return_rdps = False
        self.add_fusion = add_fusion
        if return_layers is None:
            return_layers = []
        else:
            if self.add_fusion:
                self.fusion_layers = nn.ModuleList([
                    SqueezeAndExciteFusionAdd(channels=embed_dim, return_rgb_depth_p=return_rdps)
                    for _ in range(len(return_layers))
                ])
                self.return_rdps = return_rdps
        self.return_layers = return_layers

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def interpolate_pos_encoding(self, x, w, h):
        npatch = x.shape[1] - 1
        N = self.pos_embed.shape[1] - 1
        if npatch == N and w == h:
            return self.pos_embed
        class_pos_embed = self.pos_embed[:, 0]
        patch_pos_embed = self.pos_embed[:, 1:]
        dim = x.shape[-1]
        w0 = w // self.patch_embed.patch_size
        h0 = h // self.patch_embed.patch_size
        # we add a small number to avoid floating point error in the interpolation
        # see discussion at https://github.com/facebookresearch/dino/issues/8
        w0, h0 = w0 + 0.1, h0 + 0.1
        patch_pos_embed = nn.functional.interpolate(
            patch_pos_embed.reshape(1, int(math.sqrt(N)), int(math.sqrt(N)), dim).permute(0, 3, 1, 2),
            scale_factor=(w0 / math.sqrt(N), h0 / math.sqrt(N)),
            mode='bicubic',
        )
        assert int(w0) == patch_pos_embed.shape[-2] and int(h0) == patch_pos_embed.shape[-1]
        patch_pos_embed = patch_pos_embed.permute(0, 2, 3, 1).view(1, -1, dim)
        return torch.cat((class_pos_embed.unsqueeze(0), patch_pos_embed), dim=1)

    def prepare_tokens(self, x):
        B, nc, w, h = x.shape
        x = self.patch_embed(x)  # patch linear embedding

        # add the [CLS] token to the embed patch tokens
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)

        # add positional encoding to each token
        x = x + self.interpolate_pos_encoding(x, w, h)

        return self.pos_drop(x)

    def forward(self, x, merge_layers=None):
        x = self.prepare_tokens(x)
        xs = []
        rdps = []
        j = 0
        for i, blk in enumerate(self.blocks):
            x = blk(x)
            if merge_layers is not None:
                if i in self.return_layers and self.add_fusion:
                    bs, c, dims = x.shape
                    x = self.fusion_layers[j](x.reshape(-1, dims, 1, 1),
                                              merge_layers[j].reshape(-1, dims, 1, 1)).reshape(bs, c, dims)
                    if isinstance(x, tuple):
                        x, rdp = x
                        rdps.append(rdp)
                    j += 1
            if i in self.return_layers:
                xs.append(x)

        x = self.norm(x)
        if len(xs) > 0:
            xs.append(x[:, 0])
            if self.return_rdps:
                return (xs, rdps)
            else:
                return xs
        else:
            return x[:, 0]

    def get_last_selfattention(self, x):
        x = self.prepare_tokens(x)
        for i, blk in enumerate(self.blocks):
            if i < len(self.blocks) - 1:
                x = blk(x)
            else:
                # return attention of the last block
                return blk(x, return_attention=True)

    def get_intermediate_layers(self, x, n=1):
        x = self.prepare_tokens(x)
        # we return the output tokens from the `n` last blocks
        output = []
        for i, blk in enumerate(self.blocks):
            x = blk(x)
            if len(self.blocks) - i <= n:
                output.append(self.norm(x))
        return output


def _no_grad_trunc_normal_(tensor, mean, std, a, b):
    # Cut & paste from PyTorch official master until it's in a few official releases - RW
    # Method based on https://people.sc.fsu.edu/~jburkardt/presentations/truncated_normal.pdf
    def norm_cdf(x):
        # Computes standard normal cumulative distribution function
        return (1. + math.erf(x / math.sqrt(2.))) / 2.

    if (mean < a - 2 * std) or (mean > b + 2 * std):
        warnings.warn("mean is more than 2 std from [a, b] in nn.init.trunc_normal_. "
                      "The distribution of values may be incorrect.",
                      stacklevel=2)

    with torch.no_grad():
        # Values are generated by using a truncated uniform distribution and
        # then using the inverse CDF for the normal distribution.
        # Get upper and lower cdf values
        l = norm_cdf((a - mean) / std)
        u = norm_cdf((b - mean) / std)

        # Uniformly fill tensor with values from [l, u], then translate to
        # [2l-1, 2u-1].
        tensor.uniform_(2 * l - 1, 2 * u - 1)

        # Use inverse cdf transform for normal distribution to get truncated
        # standard normal
        tensor.erfinv_()

        # Transform to proper mean, std
        tensor.mul_(std * math.sqrt(2.))
        tensor.add_(mean)

        # Clamp to ensure it's in the proper range
        tensor.clamp_(min=a, max=b)
        return tensor


def trunc_normal_(tensor, mean=0., std=1., a=-2., b=2.):
    # type: (Tensor, float, float, float, float) -> Tensor # type: ignore
    return _no_grad_trunc_normal_(tensor, mean, std, a, b)


def vit_small(patch_size=16, **kwargs):
    model = VisionTransformer(
        patch_size=patch_size, embed_dim=384, depth=12, num_heads=6, mlp_ratio=4,
        qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model


class DinoRGBWrapper(nn.Module):
    def __init__(self, arch, return_layers, down_sample_last_layer=False, num_classes=0,
                 resize_layers=True):
        super().__init__()
        self.arch = arch
        self.num_register_tokens = 0
        try:
            self.num_register_tokens = self.arch.num_register_tokens
        except:
            pass

        
        self.num_cls_tokens = 1
        try:
            self.num_cls_tokens = self.arch.num_cls_tokens
        except:
            pass           

        self.patch_size = self.arch.patch_embed.patch_size
        if not isinstance(self.patch_size, tuple):
            self.patch_size = (self.patch_size, self.patch_size)
        self.out_channels = self.arch.embed_dim
       
        self.return_layers = return_layers
        self.down_sample_last_layer = down_sample_last_layer
        self.num_classes = num_classes
        self.resize_layers = resize_layers
        self.fc = None
        if self.num_classes > 0 and not self.return_layers:
            self.fc = nn.Linear(self.out_channels, num_classes)
            self.out_channels = num_classes  
        self.interm_channels = [self.out_channels, self.out_channels, self.out_channels, self.out_channels]


    def forward(self, x):
        
        # x = x[:, :2] # check the rgb part for loading

        _, _, h, w = x.shape
        #print(x.shape)
        # debug
        # print('height of the image is : ', h)
        # print('width of the image is : ', w)
        #print('shape of patch-size is ', self.patch_size[0], self.patch_size[1])
        
        h_, w_ = h / self.patch_size[0], w / self.patch_size[1]
        
        #print('height of the image after dividing by patch size : ', h_)
        #print('width of the image after dividing by patch size : ', w_)
        
        h_decode, w_decode = ((h)//32)*32 , ((w)//32)*32 # decoder head of mask2former expects dims to be multiple of 32
        
        #print(h_, w_, h_decode, w_decode)
        # debug
        # print('height of the image after rounding of to 32 : ', h_decode)
        # print('width of the image after rounding of to 32 : ', w_decode)
        
        xs = self.arch(x)
        # debug 
        #print(f'type of xs:{type(xs)}')
        # print(f'Content of xs:{xs}') # xs is a dict with keys: return and fusion


        if isinstance(xs, dict):
            xs_return = xs.get('return',[])
            xs_fusion = xs.get('fusion',[])
            
        else:
            xs_return = xs
            xs_fusion= xs


        if self.return_layers and self.resize_layers:
            for i, x_ in enumerate(xs_return[:-1]):
                bs, l, d = x_.shape
                l = l - self.num_cls_tokens - self.num_register_tokens
                x_ = x_[:, self.num_cls_tokens+self.num_register_tokens:].reshape(bs, int(l / w_), int(l / h_), -1).permute(0, 3, 1, 2)
                t_size = (int(h_decode / (2**(i+1))), int(w_decode / (2**(i+1))))
                xs_return[i] = F.interpolate(x_, t_size, None, 'bilinear', None, recompute_scale_factor=None)
                #print(i, t_size)
            
            if self.down_sample_last_layer:
                t_size = (int(h_decode / (2 ** (i + 2))), int(w_decode / (2 ** (i + 2))))
                x_ = F.interpolate(x_, t_size, None, 'bilinear', None, recompute_scale_factor=None)
                xs_return = xs_return[:-1] + [x_] + [xs_return[-1]]
                
            xs_return = [None] + xs_return

        if self.return_layers:
            return xs_return
        else:
            out = xs_return[-1]
            if self.fc is not None:
                out = self.fc(out)
            return out
        

class DinoRGBDWrapper(nn.Module):
    def __init__(self, color_encoder, depth_encoder, return_layers, down_sample_last_layer=True, return_rdps=False,
                 cat_outs=False, resize_layers=True, rand_disable_modality=False):
        super().__init__()
        self.color_encoder = color_encoder
        self.depth_encoder = depth_encoder

        self.num_register_tokens = 0
        try:
            self.num_register_tokens = self.color_encoder.num_register_tokens
        except Exception as e:
            pass
       
        self.num_cls_tokens = 1
        try:
            self.num_cls_tokens = self.color_encoder.num_cls_tokens
        except:
            pass       

            
        self.return_layers = return_layers
        self.patch_size = self.color_encoder.patch_embed.patch_size
        if not isinstance(self.patch_size, tuple):
            self.patch_size = (self.patch_size, self.patch_size)
        self.out_channels = self.color_encoder.embed_dim
        self.down_sample_last_layer = down_sample_last_layer
        self.return_rdps = return_rdps
        self.resize_layers = resize_layers
        self.rand_disable_modality = rand_disable_modality
        self.cat_outs = cat_outs
        if self.cat_outs:
            self.out_channels = self.out_channels * 2

        self.interm_channels = [self.out_channels, self.out_channels, self.out_channels, self.out_channels]

    def train_depth_cls_token(self):
        self.depth_encoder.cls_token.requires_grad = True
        self.depth_encoder.norm.requires_grad = True

    def forward(self, x, xd, depth_scales=None):
        _, _, h, w = x.shape
        h_, w_ = h / self.patch_size[0], w / self.patch_size[1]
        h_decode, w_decode = (h // 32)*32, (w // 32)*32
        
        #print('input x', x.shape, xd.shape)
        xs = self.color_encoder(x)

        if isinstance(xs, dict):
            xs_return = xs['return']
            xs = xs['fusion']
        else:
            xs_return = xs

        # print('xd shape inside Dinorgbdwrapper dino.py', xd.shape)
        #for x_ in xs:
        #    if x_ is not None:
        #        print('x_', x_.shape)

        if len(xd.shape) == 3:
            xd = xd.unsqueeze(1)

        xds = self.depth_encoder(xd, xs[:-1], depth_scales)

        rdps = None
        if self.return_rdps:
            xds, rdps = xds
            #print('her rdps', rdps)

        if isinstance(xds, dict):
            xds_return = xds['return']
            xds = xds['fusion']
        else:
            xds_return = xds

        if self.return_layers and self.resize_layers:
            # this is for Unet Decoder in Vanishing Depth
            for i, x_ in enumerate(xds[:-1]):
                bs, l, d = x_.shape
                l = l - self.num_register_tokens
                if self.num_cls_tokens == 0 and depth_scales is not None:
                    num_cls_tokens = 1
                else:
                    num_cls_tokens = self.num_cls_tokens
                l = l - self.num_cls_tokens

                x_ = x_[:, num_cls_tokens+self.num_register_tokens:].reshape(bs, int(l / w_), int(l / h_), -1).permute(0, 3, 1, 2)
                t_size = (int(h_decode / (2**(i+1))), int(w_decode / (2**(i+1))))
                xds[i] = F.interpolate(x_, t_size, None, 'bilinear', None, recompute_scale_factor=None)

                #xds[xi] = self.upscale_factors[i](x_)
                if self.cat_outs:
                    #print('this', x_.shape, xs[i].shape, t_size)
                    #input()
                    xs_ = xs[i][:, self.num_cls_tokens+self.num_register_tokens:].reshape(bs, int(l / w_), int(l / h_), -1).permute(0, 3, 1, 2)
                    xs_ = F.interpolate(xs_, t_size, None, 'bilinear', None, recompute_scale_factor=None)
                    #print('those', xs_[i].shape, xds[i].shape)
                    xds[i] = torch.cat([xs_, xds[i]], dim=1)
                    #print('there', xds[i].shape)
                    #input()
                    
            if self.down_sample_last_layer:
                t_size = (int(h_decode / (2 ** (i + 2))), int(w_decode / (2 ** (i + 2))))
                x_ = F.interpolate(x_, t_size, None, 'bilinear', None, recompute_scale_factor=None)
                xds = xds[:-1] + [x_] + [xds[-1]]
                
                #if self.cat_outs:
                #    xs_ = xs[i][:, :-1].reshape(bs, int((l-1) / w_), int((l-1) / h_), -1).permute(0, 3, 1, 2)
                #    xs_ = F.interpolate(xs_, t_size, None, 'bilinear', None, recompute_scale_factor=None)
                #    xds[i] = torch.cat([xs_, xds[i]], dim=1)

            xds = [None] + xds
          
            if self.return_rdps:
                #print('here')
                return (xds, rdps)
            else:
                #print('there')
                return xds

        else:
            if self.cat_outs:
                for i, (x_, xd_) in enumerate(zip(xs_return, xds_return)):
                    if self.rand_disable_modality:

                        inds = torch.randint(3, (1, len(x_)))
                        #print('a', i, inds, torch.std(x_), torch.std(xd_))
                        x_[inds[0] == 1] = 0
                        xd_[inds[0] == 2] = 0
                        #print('b', i, inds, torch.std(x_), torch.std(xd_))

                    xds_return[i] = torch.cat([x_, xd_], dim=-1)
                    #print('here', i, xds_return[i].shape, x_.shape, xd_.shape)

            if self.return_rdps and rdps is not None:
                xds_return = (xds_return, rdps)

            return xds_return

        '''                  
        else:
            if isinstance(xds, list):
                xds, xs = xds[-1], xs[-1]

            if self.cat_outs:
                out = torch.cat([xds.unsqueeze(-1), xs.unsqueeze(-1)], dim=-1).flatten(1)
            else:
                out = xds

            if self.return_rdps:
                #print('here?')
                return (out, rdps)
            else:
                #print('there?')
                return out
        '''


def get_dinov1_encoder(args, vit_return_layers=None):
    if vit_return_layers is None:
        vit_return_layers = [2, 5, 8, 11]
    vit_rgb = vit_small(patch_size=16, return_layers=vit_return_layers, img_size=[224])
    if isinstance(args.pretrained_path, str):
        sd = torch.load(args.pretrained_path, map_location='cpu')
        vit_rgb.load_state_dict(sd)
        print('loaded state dict from: {}'.format(args.pretrained_path))
    vit_depth = vit_small(patch_size=16, in_chans=args.depth_channels,
                          return_layers=vit_return_layers,
                          return_rdps=args.return_rdps,
                          add_fusion=True, img_size=[224])
    model = DinoRGBDWrapper(vit_rgb, vit_depth, return_layers=vit_return_layers, return_rdps=args.return_rdps)
    return model


def get_dinov2_encoder(args, return_layers=True, vit_return_layers=None, rgb_only=False, cat_outs=False,
                       fusion_layer_ids=None, resize_layers=True, down_sample_last_layer=True,
                       norm_return_layers=False, return_before_fusion=None,
                       norm_return_layers_rgb=False, rand_disable_modality=False,
                       add_depth_scales_to_cls_token=False, dino_size='small', fuse_cls_tolken=False):

    dino_encoder = {
        'small': dinov2_small, 
        'base': dinov2_base,
        'large': dinov2_large
    }[dino_size]


    if fusion_layer_ids is None:
        fusion_layer_ids = {
            'small': [2, 5, 8, 11],
            'base': [2, 5, 8, 11],
            'large': [5, 12, 18, 24]
        }[dino_size]

    if vit_return_layers is None:
        vit_return_layers = fusion_layer_ids

    if return_before_fusion is None:
        return_before_fusion = cat_outs
    #if not return_layers:
    #    vit_return_layers = False

    print('fusion_layer_ids', fusion_layer_ids)
    print('vit_return_layers', vit_return_layers)
    print('return_before_fusion', return_before_fusion)
    print('cat_outs', cat_outs)
    print('norm_return_layers', norm_return_layers)
    print('norm_return_layers_rgb', norm_return_layers_rgb)
    print('rand_disable_modality', rand_disable_modality)
    print('add_depth_scales_to_cls_token', add_depth_scales_to_cls_token)
    print('dino_encoder', dino_encoder)
    print('fuse_cls_tolken', fuse_cls_tolken)


    vit_rgb = dino_encoder(patch_size=14,
                           return_layers=vit_return_layers,
                           fusion_layer_ids=fusion_layer_ids,
                           img_size=518, init_values=1.0,
                           block_chunks=0, norm_return_layers=norm_return_layers_rgb)
    print('return_rdps', args.return_rdps)

    if not rgb_only:
        depth_channels = args.depth_channels
        try:
            if args.with_intr:
                depth_channels = depth_channels * 3
        except:
            pass
        vit_depth = dino_encoder(patch_size=14,
                                 in_chans=depth_channels,
                                 return_layers=vit_return_layers,
                                 return_rdps=args.return_rdps,
                                 fusion_layer_ids=fusion_layer_ids,
                                 add_fusion=True, img_size=518, init_values=1.0, block_chunks=0,
                                 norm_return_layers=norm_return_layers,
                                 return_before_fusion=return_before_fusion,
                                 add_depth_scales_to_cls_token=add_depth_scales_to_cls_token,
                                 fuse_cls_tolken=fuse_cls_tolken)
    else:
        print('RGB Only encoder')
    #args.pretrained_path = './data/backbones/dinov2_vits14_pretrain.pth'
    # print('args.pretrained_path', args.pretrained_path)
    if isinstance(args.pretrained_path, str):
        if os.path.exists(args.pretrained_path):
            sd = torch.load(args.pretrained_path, map_location='cpu')   
            vit_rgb = load_fitting_state_dict(vit_rgb, sd)
            print('loaded state dict on vit rgb from: {}'.format(args.pretrained_path))
            if not rgb_only:
                vit_depth = load_fitting_state_dict(vit_depth, sd)
                #vit_depth.load_state_dict(sd)
                print('loaded state dict on vit depth from: {}'.format(args.pretrained_path))
        #input()

    if not rgb_only:
        print('DinoRGBDWrapper')
        model = DinoRGBDWrapper(vit_rgb, vit_depth, return_layers=return_layers, return_rdps=args.return_rdps,
                                cat_outs=cat_outs, resize_layers=resize_layers,
                                down_sample_last_layer=down_sample_last_layer,
                                rand_disable_modality=rand_disable_modality)
    else:
        print('DinoRGBWrapper')
        model = DinoRGBWrapper(vit_rgb, return_layers=return_layers,
                               resize_layers=resize_layers, down_sample_last_layer=down_sample_last_layer)
    return model


def get_dinov3_encoder(args, return_layers=True, vit_return_layers=None, rgb_only=False, cat_outs=False,
                       fusion_layer_ids=None, resize_layers=True, down_sample_last_layer=True,
                       norm_return_layers=False, return_before_fusion=None,
                       norm_return_layers_rgb=False, rand_disable_modality=False,
                       add_depth_scales_to_cls_token=False, dino_size='small', fuse_cls_tolken=False):

    dino_encoder = {
        'small': dinov3_small, 
        'base': dinov3_base,
        'large': dinov3_large
    }[dino_size]


    if fusion_layer_ids is None:
        fusion_layer_ids = {
            'small': [2, 5, 8, 11],
            'base': [2, 5, 8, 11],
            'large': [5, 12, 18, 24]
        }[dino_size]

    if vit_return_layers is None:
        vit_return_layers = fusion_layer_ids

    if return_before_fusion is None:
        return_before_fusion = cat_outs
    #if not return_layers:
    #    vit_return_layers = False

    print('fusion_layer_ids', fusion_layer_ids)
    print('vit_return_layers', vit_return_layers)
    print('return_before_fusion', return_before_fusion)
    print('cat_outs', cat_outs)
    print('norm_return_layers', norm_return_layers)
    print('norm_return_layers_rgb', norm_return_layers_rgb)
    print('rand_disable_modality', rand_disable_modality)
    print('add_depth_scales_to_cls_token', add_depth_scales_to_cls_token)
    print('dino_encoder', dino_encoder)
    print('fuse_cls_tolken', fuse_cls_tolken)

    vit_rgb = dino_encoder(
        return_layers=vit_return_layers,
        fusion_layer_ids=fusion_layer_ids,
        layerscale_init=1e-5, mask_k_bias=True,  n_storage_tokens=4,
        norm_return_layers=norm_return_layers_rgb
    )

    print('return_rdps', args.return_rdps)

    if not rgb_only:
        depth_channels = args.depth_channels
        try:
            if args.with_intr:
                depth_channels = depth_channels * 3
        except:
            pass

        vit_depth = dino_encoder(
            in_chans=depth_channels,
            return_layers=vit_return_layers,
            return_rdps=args.return_rdps,
            fusion_layer_ids=fusion_layer_ids,
            add_fusion=True, 
            layerscale_init=1e-5, mask_k_bias=True, n_storage_tokens=4,
            norm_return_layers=norm_return_layers,
            return_before_fusion=return_before_fusion,
            add_depth_scales_to_cls_token=add_depth_scales_to_cls_token,
            fuse_cls_tolken=fuse_cls_tolken
        )
    else:
        print('RGB Only encoder')
        
    if isinstance(args.pretrained_path, str):
        if os.path.exists(args.pretrained_path):
            sd = torch.load(args.pretrained_path, map_location='cpu')   
            vit_rgb = load_fitting_state_dict(vit_rgb, sd)
            print('loaded state dict on vit rgb from: {}'.format(args.pretrained_path))
            if not rgb_only:
                vit_depth = load_fitting_state_dict(vit_depth, sd)
                #vit_depth.load_state_dict(sd)
                print('loaded state dict on vit depth from: {}'.format(args.pretrained_path))
        #input()

    if not rgb_only:
        print('DinoRGBDWrapper')
        model = DinoRGBDWrapper(vit_rgb, vit_depth, return_layers=return_layers, return_rdps=args.return_rdps,
                                cat_outs=cat_outs, resize_layers=resize_layers,
                                down_sample_last_layer=down_sample_last_layer,
                                rand_disable_modality=rand_disable_modality)
    else:
        print('DinoRGBWrapper')
        model = DinoRGBWrapper(vit_rgb, return_layers=return_layers,
                               resize_layers=resize_layers, down_sample_last_layer=down_sample_last_layer)
    return model


class DINOHead(nn.Module):
    def __init__(self, in_dim, out_dim, use_bn=False, norm_last_layer=True, nlayers=3, hidden_dim=2048,
                 bottleneck_dim=256):
        super().__init__()
        nlayers = max(nlayers, 1)
        if nlayers == 1:
            self.mlp = nn.Linear(in_dim, bottleneck_dim)
        else:
            layers = [nn.Linear(in_dim, hidden_dim)]
            if use_bn:
                layers.append(nn.BatchNorm1d(hidden_dim))
            layers.append(nn.GELU())
            for _ in range(nlayers - 2):
                layers.append(nn.Linear(hidden_dim, hidden_dim))
                if use_bn:
                    layers.append(nn.BatchNorm1d(hidden_dim))
                layers.append(nn.GELU())
            layers.append(nn.Linear(hidden_dim, bottleneck_dim))
            self.mlp = nn.Sequential(*layers)
        self.apply(self._init_weights)
        self.last_layer = nn.utils.weight_norm(nn.Linear(bottleneck_dim, out_dim, bias=False))
        self.last_layer.weight_g.data.fill_(1)
        if norm_last_layer:
            self.last_layer.weight_g.requires_grad = False

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.mlp(x)
        x = nn.functional.normalize(x, dim=-1, p=2)
        x = self.last_layer(x)
        return x

'''
Copied and edidted from:

https://github.com/facebookresearch/dino/blob/7c446df5b9f45747937fb0d72314eb9f7b66930a/main_dino.py#L363

'''


if __name__ == '__main__':
    vit_return_layers = [2, 5, 8, 11]
    depth_channels = 16
    return_rdps = False
    upscale_factors = [8, 4, 2, 1]

    h = w = 448

    img_size = [518]
    patch_size = 14
    vit_rgb = dinov2_small(patch_size=patch_size, img_size=img_size[0], init_values=1.0, block_chunks=0,
                           return_layers=vit_return_layers).cuda()
    
    # vit_rgb = dinov2_base(patch_size=patch_size, img_size=img_size[0], init_values=1.0, block_chunks=0,
    #                        return_layers=vit_return_layers).cuda()
    
    #sd = torch.load('/home/chowanki/git/vanishing-depth-self-supervised/data/backbones/dinov2_vits14_pretrain.pth',
                    #map_location='cuda')
    #vit_rgb.load_state_dict(sd)


    #vit_rgb = vit_small(patch_size=patch_size, return_layers=vit_return_layers, img_size=img_size)
    #sd = torch.load('/home/kochpaul/git/vanishing-depth-self-supervised/data/backbones/dino_deitsmall16_pretrain.pth',
    #                map_location='cpu')
    #print(sd.keys())
    #vit_rgb.load_state_dict(sd)

    # vit_depth = dinov2_base(patch_size=patch_size, in_chans=depth_channels,
    #                          return_layers=vit_return_layers,
    #                          return_rdps=return_rdps,
    #                          add_fusion=True, img_size=img_size[0], init_values=1.0, block_chunks=0)

    vit_depth = dinov2_small(patch_size=patch_size, in_chans=depth_channels,
                             return_layers=vit_return_layers,
                             return_rdps=return_rdps,
                             add_fusion=True, img_size=img_size[0], init_values=1.0, block_chunks=0)

    model = DinoRGBDWrapper(vit_rgb, vit_depth, return_layers=True)
    model = model.cuda()

    optimizer = torch.optim.Adam(params=model.parameters(), weight_decay=0)
    bs = 8
    x = torch.rand((bs, 3, h, w)).cuda()
    xd = torch.rand((bs, depth_channels, h, w)).cuda()

    out = model(x, xd)
    for o in out:
        if o is None:
            continue
        print(o.shape)

    loss = torch.mean(out[-1])
    print(loss)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    