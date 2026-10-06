# ------------------------------------------------------------------------
# Deformable DETR
# Copyright (c) 2020 SenseTime. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
# Modified from DETR (https://github.com/facebookresearch/detr)
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
# ------------------------------------------------------------------------

"""
Backbone modules.
"""
from collections import OrderedDict
import re
from typing import Dict, List
import torch
import torch.nn.functional as F
import torchvision
from torch import nn
from torchvision.models._utils import IntermediateLayerGetter
from torchvision.models.feature_extraction import create_feature_extractor
from typing import Dict, List
from model.deformable_detr.util.misc import NestedTensor, is_main_process
from model.deformable_detr.position_encoding import build_position_encoding
from model.resnet import ResNetRGBD, ResNetv2RGBD
from model.efficientnet import EfficientNetRGBD, EfficientNetRGBDv2
from dataclasses import dataclass
from typing import Optional, Tuple
import math


@dataclass
class DepthEncodingConfig:
    mode: str = "none"                 # "posenc" | "disparity" | "none"
    apply_norm: bool = False           # normalize depth input
    depth_channels: int = 32           # used by posenc
    temperature: float = 0.0003       # used by posenc
    scale: float = 2.0 * math.pi       # used by posenc
    zero_eps: float = 1e-6             # replace 0 with eps
    # normalize mode
    mean: float = 5.0
    std: float = 5.0
    # disparity mode
    baseline: float = 0.05
    focal_length: Optional[float] = None   # if None, fallback: min(H,W)*0.9 (in pixels)
    min_depth: float = 0.00
    max_depth: float = 15.0
    norm_with_max_scale: bool = True
    rescale_factor: float = 75.0
    clamp_max_depth: bool = False
    clamp_min_depth: bool = False


class DepthEncoder(nn.Module):
    def __init__(self, cfg: DepthEncodingConfig):
        super().__init__()
        self.cfg = cfg
        self.register_buffer("_dim_t", self._make_dim_t(cfg), persistent=False)

    def _make_dim_t(self, cfg: DepthEncodingConfig) -> torch.Tensor:
        if cfg.mode != "posenc":
            return torch.empty(0)
        dim_t = torch.arange(cfg.depth_channels, dtype=torch.float32)
        dim_t = (2 * torch.div(dim_t, 2, rounding_mode='trunc')) / cfg.depth_channels
        return cfg.temperature ** dim_t

    def forward(self, d: torch.Tensor, intr: Optional[Tuple[float,float,float,float]] = None) -> torch.Tensor:
        # d: [B,1,H,W] or [B,H,W]
        if d is None:
            return None
        if d.dim() == 3:
            d = d.unsqueeze(1)  # [B,1,H,W]

        B, _, H, W = d.shape
        cfg = self.cfg

        # replace zeros for safe division/log/etc
        if cfg.zero_eps > 0:
            zmask = (d <= 0)
            d = d.clone()
            d[zmask] = cfg.zero_eps

        # encode by mode
        if cfg.mode == "posenc":
            # sinusoidal encoding on depth
            dim_t = self._dim_t.to(d.device)

            d = d.clamp(min=cfg.min_depth, max=cfg.max_depth)

            if cfg.norm_with_max_scale:
                d = d.div(cfg.max_depth)
            
            if cfg.scale > 0:
                d = d * cfg.scale
            
            # [B,1,H,W,1] / [C] => [B,H,W,C]
            d = d.permute(0, 2, 3, 1)  # [B,H,W,1]
            d = d / dim_t  # broadcast to [B,H,W,C]
            d = torch.stack((d[..., 0::2].sin(), d[..., 1::2].cos()), dim=-1).flatten(-2)  # [B,H,W,C]
            d = d.permute(0, 3, 1, 2)  # [B,C,H,W]
            
        elif cfg.mode == "disparity":
            # disp = baseline * focal / depth
            if intr is not None and cfg.focal_length is None:
                # intr tuple assumed cx, cy, fx, fy
                focal = float(intr[2])
            else:
                focal = cfg.focal_length if cfg.focal_length is not None else float(min(H, W) * 0.9)

            if cfg.min_depth is not None:
                d = d.clamp(min=cfg.min_depth)
            if cfg.max_depth is not None:
                d = d.clamp(max=cfg.max_depth)
            d = (cfg.baseline * focal) / d
            if cfg.rescale_factor and cfg.rescale_factor > 0:
                d = d / cfg.rescale_factor        

        if cfg.apply_norm == "normalize" and cfg.mode != "posenc":
            d = (d - cfg.mean) / (cfg.std if cfg.std != 0 else 1.0)  # [B,1,H,W]
    
        return d


class FrozenBatchNorm2d(torch.nn.Module):
    """
    BatchNorm2d where the batch statistics and the affine parameters are fixed.

    Copy-paste from torchvision.misc.ops with added eps before rqsrt,
    without which any other models than torchvision.models.resnet[18,34,50,101]
    produce nans.
    """

    def __init__(self, n, eps=1e-5):
        super(FrozenBatchNorm2d, self).__init__()
        self.register_buffer("weight", torch.ones(n))
        self.register_buffer("bias", torch.zeros(n))
        self.register_buffer("running_mean", torch.zeros(n))
        self.register_buffer("running_var", torch.ones(n))
        self.eps = eps

    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict,
                              missing_keys, unexpected_keys, error_msgs):
        num_batches_tracked_key = prefix + 'num_batches_tracked'
        if num_batches_tracked_key in state_dict:
            del state_dict[num_batches_tracked_key]

        super(FrozenBatchNorm2d, self)._load_from_state_dict(
            state_dict, prefix, local_metadata, strict,
            missing_keys, unexpected_keys, error_msgs)

    def forward(self, x):
        # move reshapes to the beginning
        # to make it fuser-friendly
        w = self.weight.reshape(1, -1, 1, 1)
        b = self.bias.reshape(1, -1, 1, 1)
        rv = self.running_var.reshape(1, -1, 1, 1)
        rm = self.running_mean.reshape(1, -1, 1, 1)
        eps = self.eps
        scale = w * (rv + eps).rsqrt()
        bias = b - rm * scale
        return x * scale + bias


class BackboneBase(nn.Module):
    def __init__(self, backbone: nn.Module, train_backbone: bool, return_interm_layers: bool):
        super().__init__()
        for name, parameter in backbone.named_parameters():
            if not train_backbone or 'layer2' not in name and 'layer3' not in name and 'layer4' not in name:
                parameter.requires_grad_(False)

        if return_interm_layers:
            # return_layers = {"layer1": "0", "layer2": "1", "layer3": "2", "layer4": "3"}
            return_layers = {"layer2": "0", "layer3": "1", "layer4": "2"}
            self.strides = [8, 16, 32]
            self.num_channels = [512, 1024, 2048]
        else:
            return_layers = {'layer4': "0"}
            self.strides = [32]
            self.num_channels = [2048]
        self.body = IntermediateLayerGetter(backbone, return_layers=return_layers)

    def forward(self, tensor_list: NestedTensor):
        xs = self.body(tensor_list.tensors)
        out: Dict[str, NestedTensor] = {}
        for name, x in xs.items():
            m = tensor_list.mask
            assert m is not None
            mask = F.interpolate(m[None].float(), size=x.shape[-2:]).to(torch.bool)[0]
            out[name] = NestedTensor(x, mask)
        return out


class ResNetWrapper(nn.Module):
    def __init__(self, backbone: nn.Module, return_interm_layers: bool = True):
        super().__init__()

        
        return_layers = {"layer1": "0", "layer2": "1", "layer3": "2", "layer4": "3"}
        self.num_channels = [256, 512, 1024, 2048]
        self.body = create_feature_extractor(backbone.arch, return_nodes=return_layers)
        #self.body = IntermediateLayerGetter(backbone.arch, return_layers=return_layers)

    def forward(self, tensor_list: NestedTensor):
        xs = self.body(tensor_list.tensors)
        out: Dict[str, NestedTensor] = {}
        for name, x in xs.items():
            m = tensor_list.mask
            assert m is not None
            mask = F.interpolate(m[None].float(), size=x.shape[-2:]).to(torch.bool)[0]
            #print(name, x.shape, mask.shape)
            out[name] = NestedTensor(x, mask)
        return out




class Backbone(BackboneBase):
    """ResNet backbone with frozen BatchNorm."""
    def __init__(self, name: str,
                 train_backbone: bool,
                 return_interm_layers: bool,
                 dilation: bool):
        norm_layer = FrozenBatchNorm2d
        backbone = getattr(torchvision.models, name)(
            replace_stride_with_dilation=[False, False, dilation],
            pretrained=is_main_process(), norm_layer=norm_layer)
        assert name not in ('resnet18', 'resnet34'), "number of channels are hard coded"
        super().__init__(backbone, train_backbone, return_interm_layers)
        if dilation:
            self.strides[-1] = self.strides[-1] // 2


class BackboneBaseRGBD(nn.Module):
    '''
    Modified Backbone for RGB-D based on the RGB version (BackboneBase)
    '''

    def __init__(self, model: nn.Module, return_interm_layers: bool, num_feature_levels: int):
        super().__init__()

        self.model = model
        self.return_interm_layers = return_interm_layers
        self.num_feature_levels = num_feature_levels
        self.num_channels = list(model.channels)
        self.l = len(self.num_channels) - self.num_feature_levels + 1
        self.num_channels = self.num_channels[self.l:]#[-(self.num_feature_levels-1):]
        if not return_interm_layers:
            self.num_channels = [self.num_channels[-1]]
            self.l = 0

    def forward(self, tensor_list: NestedTensor):
        xs = self.model(tensor_list.tensors, tensor_list.depth)
        ignore = ['input', 'out']
        if not self.return_interm_layers and len(xs) > 1:
            s_keys = sorted([key for key in xs.keys() if key not in ignore])
            if len(s_keys) > 1:
                ignore += s_keys[:-1]

        out: Dict[str, NestedTensor] = {}
        for i, (name, x) in enumerate(xs.items()):
            if i <= self.l:
                continue

            if name in ignore:
                continue
            m = tensor_list.mask
            assert m is not None
            mask = F.interpolate(m[None].float(), size=x.shape[-2:]).to(torch.bool)[0]
            out[name] = NestedTensor(x, mask)
        return out


class BackboneRGBD(BackboneBaseRGBD):
    '''
       Modified Backbone for RGB-D based on the RGB version (Backbone)
    '''

    """ResNet backbone with frozen BatchNorm."""
    def __init__(self, name: str, version: str, depth_channels: int, pretrained: bool, return_interm_layers: bool,
                 num_feature_levels: int = -1):
        if name == 'ResNet_x->xd':
            model = ResNetv2RGBD(version, 0, return_layers=return_interm_layers, depth_channels=depth_channels,
                                 pretrained=pretrained, return_as_list=False)
        elif name == 'ResNet_xd->x':
            model = ResNetRGBD(version, 0, return_layers=return_interm_layers, depth_channels=depth_channels,
                               pretrained=pretrained, return_as_list=False)
        elif name == 'EfficientNet_x->xd':
            model = EfficientNetRGBDv2(version, 0, return_layers=return_interm_layers, depth_channels=depth_channels,
                                       pretrained=pretrained, return_cbas_list=False)
        elif name == 'EfficientNet_xd->x':
            model = EfficientNetRGBD(version, 0, return_layers=return_interm_layers, depth_channels=depth_channels,
                                     pretrained=pretrained, return_as_list=False)
        else:
            raise NotImplementedError('Model "{}" not implemented'.format(name))
        super().__init__(model, return_interm_layers, num_feature_levels)



def _parse_proj_token(token: str):
    t = token.strip()
    m_u = re.search(r'(\d+)u', t)  # times of 2x deconv
    m_k = re.search(r'(\d+)k', t)  # Jk => J convs with kernel size J
    m_p = re.search(r'(\d+)p', t)  # Jk => J convs with kernel size J
    up_count = int(m_u.group(1)) if m_u else 0
    k_val = int(m_k.group(1)) if m_k else 0
    p_val = int(m_p.group(1)) if m_p else 0
    return up_count, k_val, p_val

    
def _gn_groups(ch: int):
    # pick a divisor of ch up to 32
    for g in [32, 16, 8, 4, 2, 1]:
        if ch % g == 0:
            return g
    return 1


class Joiner(nn.Sequential):
    def __init__(self, backbone, position_embedding, position_embedding_obj=None, upsampling=None, hidden_dim=None, depth_encoder=None):
        super().__init__(backbone, position_embedding, position_embedding_obj)
        self.num_channels = backbone.num_channels
        self.num_backbone_outs = len(backbone.num_channels)
        self.upsampling = upsampling
        self.hidden_dim = hidden_dim    
        self.depth_encoder = depth_encoder
        self.proj_and_up = None
        if upsampling is not None and hidden_dim is not None:
            proj_and_up = []
            specs = None
            if isinstance(upsampling, str) and upsampling.strip():
                specs = [s.strip() for s in upsampling.replace(' ', '').split(',')]
                if len(specs) < self.num_backbone_outs:
                    specs += [specs[-1]] * (self.num_backbone_outs - len(specs))
                elif len(specs) > self.num_backbone_outs:
                    specs = specs[:self.num_backbone_outs]

            for i in range(self.num_backbone_outs):
                in_ch = backbone.num_channels[i]
                blocks = []
                # 1) Always map channels -> hidden_dim first
                up_count, k_val, p_val = _parse_proj_token(specs[i])
                
                if p_val > 0:
                    blocks += [
                        nn.Conv2d(in_ch, hidden_dim, kernel_size=1, padding=0),
                        nn.GroupNorm(_gn_groups(hidden_dim), hidden_dim),
                        nn.GELU(),
                    ]
                    in_ch = hidden_dim

                # 2) Optional upsampling and post-upsampling convs                
                for _ in range(up_count):
                    blocks += [
                        nn.ConvTranspose2d(in_ch, hidden_dim, kernel_size=4, stride=2, padding=1),
                        nn.GroupNorm(_gn_groups(hidden_dim), hidden_dim),
                        nn.GELU(),
                    ]                    
                    in_ch = hidden_dim
                
                for _ in range(k_val):
                    k = max(1, k_val)
                    blocks += [
                        nn.Conv2d(in_ch, hidden_dim, kernel_size=k, padding=k // 2),
                        nn.GroupNorm(_gn_groups(hidden_dim), hidden_dim),
                        nn.GELU(),
                    ]
                    in_ch = hidden_dim
                if len(blocks) > 0:
                    proj_and_up.append(nn.Sequential(*blocks))

            if len(proj_and_up) > 1:
                self.proj_and_up = nn.ModuleList(proj_and_up)
                   # Init all convs
                for mod in self.proj_and_up:
                    for m in mod.modules():
                        if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
                            nn.init.xavier_uniform_(m.weight, gain=1.0)
                            if m.bias is not None:
                                nn.init.zeros_(m.bias)
                # After projection, every level outputs hidden_dim channels
                self.num_channels = [hidden_dim for _ in range(self.num_backbone_outs)]            

    def extract_cls_token(self, x):
        return self[0].extract_cls_token(x)

    def forward(self, nt: NestedTensor):
        if self.depth_encoder:
            nt.depth = self.depth_encoder(nt.depth)

        xs = self[0](nt)
        out: List[NestedTensor] = []
        pos = []
        pos_obj = []
        for lvl, (name, nt) in enumerate(sorted(xs.items())):
            if self.proj_and_up is not None:
                proj = self.proj_and_up[lvl](nt.tensors)  # may change spatial size
                #print('usampled & projected {} from {} -> {}'.format(lvl, nt.tensors.shape, proj.shape))
                # resize mask to proj spatial size
                mask_resized = F.interpolate(nt.mask[None].float(), size=proj.shape[-2:]).to(torch.bool)[0]
                nt_proj = NestedTensor(proj, mask_resized)
                out.append(nt_proj)
            else:
                out.append(nt)

        # position encoding
        for i, x in enumerate(out):
            pos.append(self[1](x).to(x.tensors.dtype))

        
        # position encoding
        if self[2] is not None:
            for i, x in enumerate(out):
                pos_obj.append(self[2](x).to(x.tensors.dtype))

        if len(pos_obj) == 0:
            pos_obj = None            
        return out, pos, pos_obj  # none for the not used bbox mode


class ViTBackboneRGBD(nn.Module):
    '''
    Modified Backbone for VIT RGB-D based on the RGB version (BackboneBase)
    '''

    def __init__(self, model: nn.Module, freeze_encoder=True, freeze_color_encoder=False, resize_attention_maps=False):
        super().__init__()
        self.generator = None
        if isinstance(model, tuple):
            model, self.generator = model
            print('Freezing Generator')
            for param in self.generator.parameters():
                param.requires_grad = False

        self.model = model
        self.freeze_encoder = freeze_encoder
        self.freeze_color_encoder = freeze_color_encoder
        self.resize_attention_maps = resize_attention_maps
        self.num_channels = model.interm_channels
        #print('self.num_channels', self.num_channels, model.out_channels, model.interm_channels)
        #input()

        self.patch_size = self.model.patch_size
        if isinstance(self.patch_size, int):
            self.patch_size = (self.patch_size, self.patch_size)

        if self.freeze_encoder:
            print('Freezing Encoder')
            for param in self.model.parameters():
                param.requires_grad = False
        elif self.freeze_color_encoder:
            print('Freezing Only Color Encoder')
            for param in self.model.color_encoder.parameters():
                param.requires_grad = False
        else:
            print('Backbone not frozen, getting finetuned')

        self.num_prefix_tokens = 1
        try:
            self.num_prefix_tokens = self.model.num_cls_tokens
        except Exception as e:
            pass
        try: 
            self.num_prefix_tokens += self.model.num_register_tokens
        except Exception as e:
            pass

    def extract_cls_token(self, x):
        return self.model.color_encoder.extract_cls_token(x)
      
    def forward(self, tensor_list: NestedTensor):
        if self.generator is not None:
            self.generator.eval()
            tensor_list.depth = self.generator(tensor_list.tensors, tensor_list.depth)

        if self.freeze_encoder:
            self.model.eval()
            with torch.no_grad():
                xs = self.model(tensor_list.tensors, tensor_list.depth)
        else:
            xs = self.model(tensor_list.tensors, tensor_list.depth)

        if isinstance(xs, dict):
            xs = xs['return']
        xs = xs[:-1]
        bs, _, h, w = tensor_list.tensors.shape
        h_ = h // self.patch_size[0]
        w_ = w // self.patch_size[1]
        h_out = (h // 32) * 32
        w_out = (w // 32) * 32

        out: Dict[str, NestedTensor] = {}
        for i, x in enumerate(xs):
            if x is None:
                continue
            
            if x.dim() == 3:            
                x = x[:, self.num_prefix_tokens:].permute(0, 2, 1).view(bs, -1, h_, w_)
                if self.resize_attention_maps:
                    f = 2**(i+2)
                    x = F.interpolate(x, (h_out//f, w_out//f), mode='bilinear')
                    
            elif x.shape[-2] <= 1 and x.shape[-1] <= 1:
                continue
            
            m = tensor_list.mask
            assert m is not None
            mask = F.interpolate(m[None].float(), size=x.shape[-2:]).to(torch.bool)[0]
            out[str(i)] = NestedTensor(x, mask)
        return out


class ViTBackboneRGB(nn.Module):
    '''
    Modified Backbone for VIT RGB-D based on the RGB version (BackboneBase)
    '''

    def __init__(self, model: nn.Module, freeze_encoder=True, resize_attention_maps=False):
        super().__init__()

        self.model = model
        self.freeze_encoder = freeze_encoder
        self.resize_attention_maps = resize_attention_maps
        self.num_channels = model.interm_channels
        
        self.patch_size = self.model.patch_size
        if isinstance(self.patch_size, int):
            self.patch_size = (self.patch_size, self.patch_size)

        if self.freeze_encoder:
            print('Freezing Encoder')
            for param in self.model.parameters():
                param.requires_grad = False

        self.num_prefix_tokens = 1
        try:
            self.num_prefix_tokens = self.model.num_cls_tokens
        except Exception as e:
            pass
        try: 
            self.num_prefix_tokens += self.model.num_register_tokens
        except Exception as e:
            pass

        
        
    def forward(self, tensor_list: NestedTensor):
        if self.freeze_encoder:
            self.model.eval()
            with torch.no_grad():
                xs = self.model(tensor_list.tensors)
        else:
            xs = self.model(tensor_list.tensors)

        if isinstance(xs, dict):
            xs = xs['return']
        xs = xs[:-1]
        bs, _, h, w = tensor_list.tensors.shape
        h_ = h // self.patch_size[0]
        w_ = w // self.patch_size[1]
        h_out = (h // 32) * 32
        w_out = (w // 32) * 32

        out: Dict[str, NestedTensor] = {}
        for i, x in enumerate(xs):
            if x is None:
                continue
            
            if x.dim() == 3:            
                x = x[:, self.num_prefix_tokens:].permute(0, 2, 1).view(bs, -1, h_, w_)
                
                if self.resize_attention_maps:
                    xh_, xw_ = x.shape[-2:]
                    f = 2**(i+2)
                    if h_out//f > xh_ or w_out//f > xw_:
                        x = F.interpolate(x, (h_out//f, w_out//f), mode='bilinear')
                    #print(i, xh_, xw_, f, h_out//f, w_out//f, h_out, w_out, h_, w_, x.shape)


            elif x.shape[-2] <= 1 and x.shape[-1] <= 1:
                continue

            m = tensor_list.mask
            assert m is not None
            mask = F.interpolate(m[None].float(), size=x.shape[-2:]).to(torch.bool)[0]
            out[str(i)] = NestedTensor(x, mask)
        return out


def build_backbone(args, backbone):
    position_embedding = build_position_encoding(args)

    position_embedding_obj = None
    if args.with_obj or args.with_pose or args.with_critique:
        position_embedding_obj = build_position_encoding(args, args.hidden_dim_obj)
    
    is_rgbd = False

    if args.model_name.lower() == 'resnet':
        backbone = ResNetWrapper(backbone)
    elif args.rgb_only:
        backbone = ViTBackboneRGB(backbone, resize_attention_maps=args.resize_attention_maps, freeze_encoder=args.freeze_encoder)        
    else:
        backbone = ViTBackboneRGBD(backbone, resize_attention_maps=args.resize_attention_maps, freeze_encoder=args.freeze_encoder)
        is_rgbd = True

    # depth encoding config from args
    depth_cfg = None
    depth_encoder = None

    print('-------------------------')
    print('building backbone with ', getattr(args, "with_depth", False), getattr(args, "with_positional_depth_encoding", False))
    print('-------------------------')
    if getattr(args, "with_depth", False):
        if getattr(args, "with_positional_depth_encoding", False):
            depth_cfg = DepthEncodingConfig(
                mode="posenc",
                depth_channels=getattr(args, "depth_channels", 32),
                temperature=getattr(args, "temperature", 0.0003),
                scale=getattr(args, "scale", float(2*math.pi)),
                zero_eps=getattr(args, "zero_eps", 1e-6),
                max_depth=getattr(args, "max_depth", 15.0),
            )

        elif getattr(args, "encode_disparity", False):
            depth_cfg = DepthEncodingConfig(
                mode="disparity",
                baseline=getattr(args, "baseline", 0.05),
                min_depth=getattr(args, "min_depth", 0.01),
                max_depth=getattr(args, "max_depth", 15.0),
                rescale_factor=getattr(args, "rescale_factor", 75.0),
            )
        
        
        if getattr(args, "normalize_depth", True):
            if depth_cfg is None:
                depth_cfg = DepthEncodingConfig(
                    apply_norm=True,
                    mean=getattr(args, "depth_mean", 5.0),
                    std=getattr(args, "depth_std", 5.0),
                )
            else:
                depth_cfg.apply_norm = True                
                depth_cfg.depth_mean = getattr(args, "depth_mean", 5.0)
                depth_cfg.depth_std = getattr(args, "depth_std", 5.0)
        
    if depth_cfg is not None:
        depth_encoder = DepthEncoder(depth_cfg) if args.with_depth and not args.rgb_only else None

    model = Joiner(backbone, position_embedding, position_embedding_obj, upsampling=args.backbone_proj_and_upsampling, hidden_dim=args.hidden_dim_det, depth_encoder=depth_encoder)
    return model, is_rgbd
