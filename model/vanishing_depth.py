import segmentation_models_pytorch as smp
import model.resnet as resnet
import model.efficientnet as effnets
import torch
import torch.nn as nn
import model.decoder as d
from segmentation_models_pytorch.base.heads import SegmentationHead


import os
from datasets.transforms import DepthPositionalEncoding
from model.linear_seghead import BNHead
import torch.nn.functional as F
from torchvision.transforms import InterpolationMode
import torchvision.transforms.functional as tvf
from utils.stuff import load_fitting_state_dict
from model.dino import DINOHead, get_dinov1_encoder, get_dinov2_encoder, get_dinov3_encoder

try:
    from model.fit3d import get_fit3d_encoder
except Exception as e:
    print('Warning could not load Fit3D with err: {}'.format(e))
try:
    from model.siglip2 import get_siglip2_encoder
except Exception as e:
    print('Warning could not load SigLip2 with err: {}'.format(e))
try:
    from model.omnivore.omniwrapper import OmniVoreWrapper
except Exception as e:
    print('Warning could not load Omnivore with err: {}'.format(e))
try:
    from model.EVA.eva2_wrapper import get_evav2_encoder
except Exception as e:
    print('Warning could not load Eva with err: {}'.format(e))
try:
    from model.dunewrapper import get_dune_encoder
except Exception as e:
    print('Warning could not load Dune with err: {}'.format(e))
try:
    from model.dformerwrapper import DformerWrapper, DformerV2Wrapper
except Exception as e:
    print('Warning could not load Dformer with err: {}'.format(e))
try:    
    from model.multimaewrapper import MultiMAEWrapper 
except Exception as e:
    print('Warning could not load MultiMae with err: {}'.format(e))
try:    
    from model.crocoencoderwrapper import CroCoEncoderWrapper
except Exception as e:
    print('Warning could not load Croco with err: {}'.format(e))
try:    
    from model.mast3rencoderwrapper import MASt3REncoderWrapper
except Exception as e:
    print('Warning could not load MASt3R with err: {}'.format(e))
try:    
    from model.omnidcwrapper import OmniDCWrapper
except Exception as e:
    print('Warning could not load OmniDC with err: {}'.format(e))
try:    
    from model.dmd3cwrapper import DMD3CWrapper
except Exception as e:
    print('Warning could not load DMD3C with err: {}'.format(e))
try: 
    from model.depthanythingv2wrapper import DepthAnythingV2Wrapper, DinV2andDepthAnythingV2Wrapper
except Exception as e:
    print('Warning could not load DepthAnythingV2 with err: {}'.format(e))


class VanishingDepth(nn.Module):
    def __init__(self, encoder, decoder, fpn=False, dropout: float = 0.5, with_depth: bool = True,
                 with_intr=False, decode_dists=None, decode_activation='relu', decode_dist_scale_inv_log10=False,
                 with_dino_head=False, dino_head_out_dims=65536, return_rdps=False,
                 decode_floor=1e-4, decode_floor_exp=4.0, rescale_output_shape=False):
        super(VanishingDepth, self).__init__()
        self.encoder = encoder
        self.decoder = decoder

        print('decoder', self.decoder.__class__.__name__)

        self.fpn = fpn
        if dropout > 0:
            self.dropout = nn.Dropout(p=dropout)
        else:
            self.dropout = None
        self.with_depth = with_depth
        self.decode_activation = decode_activation
        self.activation = torch.nn.LeakyReLU() if decode_activation == 'leaky-relu' else None
        self.decode_dist_scale_inv_log10 = decode_dist_scale_inv_log10
        self.num_out_dists = 3 if with_intr else 1
        self.decode_floor = decode_floor
        self.decode_floor_exp = decode_floor_exp
        self.rescale_output_shape = rescale_output_shape
        if isinstance(decode_dists, list):
            self.l_ = len(decode_dists)
            self.decode_dists = torch.tensor(decode_dists).view(1, self.l_, 1, 1, 1)
        else:
            self.l_ = 1
            self.decode_dists = None

        self.out_channels = self.num_out_dists * self.l_

        if self.fpn:
            self.depth_estimator = nn.ModuleList([SegmentationHead(
                in_channels=i,
                out_channels=self.out_channels,
                activation=None,
                kernel_size=3) for i in decoder.decoder_channels]
            )
        else:
            self.depth_estimator = SegmentationHead(
                in_channels=decoder.decoder_channels[-1],
                out_channels=self.out_channels,
                activation=None,
                kernel_size=3)
        self.with_dino_head = with_dino_head

        self.dino_head = None
        self.dino_head_out_dims = dino_head_out_dims
        if self.with_dino_head:
            self.dino_head = DINOHead(encoder.out_channels, dino_head_out_dims)
        self.return_rdps = return_rdps

        if int(str(torch.__version__).split('.')[0]) >= 2 and False:
            print('compiling decoder')
            self.decoder = torch.compile(self.decoder)
            print('compiling depth_estimator')
            self.depth_estimator = torch.compile(self.depth_estimator)
            if self.dino_head is not None and False:
                print('compiling depth_estimator')
                self.dino_head = torch.compile(self.dino_head)

    def forward(self, x, depth=None, depth_scales=None, without_decode=False):
        #if depth is not None and self.with_depth:
        #    embs = self.encoder(x, depth)
        #else:

        s = x.shape

        #print(s, depth.shape, depth_scales)

        embs = self.encoder(x, depth, depth_scales)
        if self.return_rdps:
            embs, rdps = embs

        embs, x = embs[:-1], embs[-1]
        if self.with_dino_head:
            x = self.dino_head(x)
            if without_decode:
                return x

        if self.dropout is not None and self.training:
            embs = [embs[0]] + [self.dropout(e) for e in embs[1:]]

        #print('############# there')
        #for ii, e in enumerate(embs):
        #    if e is None:
        #        print(ii, 'None')
        #    else:
        #        print(ii, e.shape)

            
        if self.decoder.__class__.__name__ == 'DPT':
            embs = self.decoder(embs, s)
        else:
            #if embs[0] is not None:
            #    embs = [None] + embs
            embs = self.decoder(embs)
            if isinstance(embs, list):
                embs = embs[1:]

        #print('############# here')
        #for ii, e in enumerate(embs):
        #    if e is None:
        #        print(ii, 'None')
        #    else:
        #        print(ii, e.shape)
                
        if self.fpn:
            out = []
            for m, depth_estimator in zip(embs, self.depth_estimator):
                dists = depth_estimator(m)
                if self.decode_dists is not None:
                    dists = self.decode_embs_2_dist(dists)

                if self.rescale_output_shape:
                    dists = F.interpolate(dists, (s[-2], s[-1]))
                
                #print(dists.shape, m.shape)
                
                out.append(dists)
        else:
            out = self.depth_estimator(embs)
            if self.decode_dists is not None:
                out = self.decode_embs_2_dist(out)

            if self.rescale_output_shape:
                    out = F.interpolate(out, (s[-2], s[-1]))
                

        if self.with_dino_head:
            out = {'maps': out, 'student_pred': x}
        if self.return_rdps:
            if isinstance(out, dict):
                out['rdps'] = rdps
            else:
                out = {'maps': out, 'rdps': rdps}
        return out

    def decode_embs_2_dist(self, x):
        bs, c, h, w = x.shape
        x = x.view(bs, self.l_, -1, h, w)
        if self.decode_activation == 'relu':
            x = torch.relu(x)
        elif self.decode_activation == 'leaky-relu':
            x = self.activation(x)
        elif self.decode_activation == 'softmax':
            x = torch.softmax(x, dim=1)
        elif self.decode_activation == 'sigmoid':
            x = torch.sigmoid(x)
        elif self.decode_activation == 'none':
            pass
        else:
            raise NotImplementedError('Decode Activation {} is not implemented'.format(self.decode_activation))

        
        floor = torch.ones(x.shape, dtype=x.dtype, device=x.device)
        floor[x < self.decode_floor] = x[x < self.decode_floor].detach().pow(self.decode_floor_exp)
        x = x.mul(floor)
        #x[x < self.decode_floor] = x[x < self.decode_floor].pow(self.decode_floor_exp)
        #print('no floor')

        decode_dists = self.decode_dists.to(x.device)
        x = x.mul(decode_dists)
        x = torch.sum(x, dim=1)
        if self.decode_dist_scale_inv_log10:
            x = torch.pow(10, x) - 1

        return x


class SegModel(nn.Module):
    def __init__(self, encoder, decoder, head, num_classes, dropout: float = 0.5, freeze_encoder=True):
        super(SegModel, self).__init__()

        self.freeze_encoder = freeze_encoder
        self.generator = None
        if isinstance(encoder, tuple):
            encoder, self.generator = encoder
            print('Freezing Generator')
            for param in self.generator.parameters():
                param.requires_grad = False
        self.encoder = encoder

        if self.freeze_encoder:
            print('Freezing Encoder')
            for param in self.encoder.parameters():
                param.requires_grad = False

        self.decoder = decoder
        self.out_channels = num_classes
        if dropout > 0:
            self.dropout = nn.Dropout(p=dropout)
        else:
            self.dropout = None

        self.seg_head = head

    def forward(self, x):
        if self.freeze_encoder:
            self.encoder.eval()

        if isinstance(x, list): # case of rgb d input
            s = x[0].shape
            if self.generator is not None:
                x, depth = x
                self.generator.eval()
                with torch.no_grad():
                    depth = self.generator(x, depth)
                if self.freeze_encoder:
                    with torch.no_grad():
                        x = self.encoder(x, depth)
                else:
                    x = self.encoder(x, depth)
            else:
                if self.freeze_encoder:
                    with torch.no_grad():
                        x = self.encoder(*x)
                else:
                    x = self.encoder(*x)
        else:
            s = x.shape
            if self.freeze_encoder:
                with torch.no_grad():
                    x = self.encoder(x)
            else:
                x = self.encoder(x)

        if isinstance(x, tuple):
            x, rdps = x
        else:
            rdps = None

        if isinstance(x, dict):
            x = x['return']

        x = x[:-1]
        for i, tensor in enumerate(x):
            if tensor is not None: 
                print(f'Encoder output tensor{i} shape:{tensor.shape}')
            else:
                print(f'Encoder output tensor{i} is None')
        
       
        if self.dropout is not None and self.training:
            x = [self.dropout(emb) for emb in x]
        x = self.decoder(x, s)
        print(f"decoder output shape:{x.shape}")
        
        x = self.seg_head(x)
        print(f"segmentation head output shape:{x.shape}, num_classes: {self.out_channels}")
        
        if rdps is not None:
            return (x, rdps)
        else:
            return x


def build_model(args):
    try:
        decoder = args.decoder
    except:
        decoder = 'UNet'

    if decoder == 'UNet':
        resize = True
    else:
        resize = False
        
    encoder, encoder_channels, decoder_channels = build_encoder(args, resize_layers=resize)

    print('decoder', decoder)
        
    print('encoder_channels', encoder_channels)
    if decoder == 'UNet':
        decoder = d.UnetDecoder(encoder_channels, decoder_channels, fpn=args.fpn, n_blocks=args.fpn_layers)

    elif 'DPT' in decoder :
        from model.dpt import DPT, DPTSegHead, DPTDepthHead
        decoder = DPT(features=encoder_channels[-1], multi_scale=args.fpn, num_register_tokens=encoder.num_register_tokens)

    elif decoder == 'linear':
        from model.linear_seghead import BNHead
        decoder = BNHead(in_channels=encoder_channels[-1], layers=args.fpn_layers, patch_size=14, out_channels=args.fpn)

    else:
        raise ValueError('Decoder {} not known'.format(args.decoder))

    if args.decode_dist:
        assert args.num_decode_dists > 0 and args.decode_max_dist > 0 and args.decode_dist_factor > 0
        decode_dists = [args.decode_max_dist]
        j = 1
        if args.decode_zero_dist:
            j = 2
        for i in range(args.num_decode_dists-j):
            decode_dists.append(decode_dists[-1] / args.decode_dist_factor)
        if args.decode_zero_dist:
            decode_dists.append(1e-8)
        print('Using {} decode dists'.format(decode_dists))
    else:
        decode_dists = None

    if args.decode_dist:
        if args.decode_dist_relu:
            decode_activation = 'relu'
        elif args.decode_dist_leaky_relu:
            decode_activation = 'leaky-relu'
        elif args.decode_dist_sigmoid:
            decode_activation = 'sigmoid'
        elif args.decode_dist_softmax:
            decode_activation = 'softmax'
        else:
            decode_activation = None
        print('Using {} as decode activation'.format(decode_activation))
        try:
            decode_dist_scale_inv_log10 = args.decode_dist_scale_inv_log10
        except:
            decode_dist_scale_inv_log10 = False
    else:
        decode_activation = None
        decode_dist_scale_inv_log10 = False

    if decode_dist_scale_inv_log10:
        print('Scaling dist output with inv log10')

    return VanishingDepth(encoder, decoder, fpn=args.fpn, dropout=args.dropout, with_depth=args.with_depth,
                          with_intr=args.with_intr, decode_dists=decode_dists, decode_activation=decode_activation,
                          decode_dist_scale_inv_log10=decode_dist_scale_inv_log10, with_dino_head=args.with_dino_head,
                          dino_head_out_dims=args.dino_head_out_dims, return_rdps=args.return_rdps,
                          decode_floor=args.decode_floor, decode_floor_exp=args.decode_floor_exp)

def build_encoder(args, return_layers=True, out_dims=0, cat_outs=False, vit_return_layers=None, with_generator=False,
                  resize_layers=True, down_sample_last_layer=True, norm_return_layers=False, fusion_layer_ids=None,
                  norm_return_layers_rgb=False, return_before_fusion=None, rand_disable_modality=False, upsample_outs=True,
                  downsample_feature_maps=False):
    
    decoder_channels = [512, 256, 128, 64, 32][:args.fpn_layers]
    #decoder_channels = [512, 512, 256, 128, 64][:args.fpn_layers]
    down_sample_last_layer = True if args.fpn_layers > 4 else False

    depth_out_dims = 0
    try:
        if args.with_dino_head:
            out_dims = 384
    except:
        pass
    if getattr(args, 'with_positional_encoding', False) or getattr(args, 'with_positional_depth_encoding', False):
        depth_channels = args.depth_channels
    else:
        depth_channels = 1
        args.depth_channels = 1

    rgb_only = False
    try:
        rgb_only = args.rgb_only
    except:
        pass

    add_depth_scales_to_cls_token = False
    try:
        add_depth_scales_to_cls_token = args.add_depth_scales_to_cls_token
    except:
        pass

    freeze_color_encoder = False
    try:
        freeze_color_encoder = args.freeze_color_encoder
    except:
        pass

    

    if getattr(args, 'pretrained_imagenet', False):
        pretrained = True
        print('using ImageNet Weights in the RGB encoding')
        cp = None
    else:
        pretrained = False
        if os.path.exists(args.pretrained_path):
            cp = torch.load(args.pretrained_path, map_location='cpu')
        elif args.pretrained_path == '':
            cp = None
            print('Using no pretrained weights')
        else:
            cp = None
            print('Pretrained path "{}" does not exist.'.format(args.pretrained_path))

    if cp is not None:
        #print('cp keys', cp.keys())
        if 'args' in cp:
            if isinstance(cp['args'], dict):
                print('Loading args from pretrained checkpoint')
                for arg, v in cp['args'].items():
                    if v != getattr(args, arg):
                        print('Setting arg {}: from {} -> {}'.format(arg, getattr(args, arg), v))
                    setattr(args, arg, v)
                    #print(arg, getattr(args, arg))

    try:
        if args.with_intr:
            depth_channels = depth_channels * 3
    except Exception as e:
        pass

    print('depth_channels', depth_channels)
    print(args.model_name, args.model_version)

    pretrained_loaded_in_get = False

    if args.model_name == 'ResNet_xd->x':
        encoder = resnet.ResNetRGBD(args.model_version, num_classes=out_dims, pretrained=pretrained,
                                    return_layers=return_layers,
                                    depth_channels=depth_channels, with_depth=args.with_depth)
        ec = resnet.__resnets_channels__[args.model_version]
        encoder_channels = (3, ec[0], ec[1], ec[2], ec[3], ec[4])
    elif args.model_name == 'ResNet_x->xd':
        encoder = resnet.ResNetv2RGBD(args.model_version, num_classes=out_dims, pretrained=pretrained,
                                      return_layers=return_layers,
                                      depth_channels=depth_channels, with_depth=args.with_depth,
                                      return_rdps=args.return_rdps,
                                      depth_classification=depth_out_dims)
        ec = resnet.__resnets_channels__[args.model_version]
        encoder_channels = (3, ec[0], ec[1], ec[2], ec[3], ec[4])
    elif args.model_name == 'ResNet_x<->xd':
        encoder = resnet.ResNetv3RGBD(args.model_version, num_classes=out_dims, pretrained=pretrained,
                                      return_layers=return_layers,
                                      depth_channels=depth_channels, with_depth=args.with_depth)
        ec = resnet.__resnets_channels__[args.model_version]
        encoder_channels = (3, ec[0], ec[1], ec[2], ec[3], ec[4])
    elif args.model_name == 'ResNet':
        return_nodes = {
            'layer1': 'layer1',
            'layer2': 'layer2',
            'layer3': 'layer3',
            'layer4': 'layer4',
        }
        encoder = resnet.ResNet(args.model_version, num_classes=0, pretrained=True, return_nodes=return_nodes)
        ec = encoder.interm_channels
        encoder_channels = (3, ec[0], ec[1], ec[2], ec[3])
                
        
    elif args.model_name == 'EfficientNet_xd->x':
        encoder = effnets.EfficientNetRGBD(args.model_version, num_classes=out_dims, pretrained=pretrained,
                                           return_layers=return_layers,
                                           depth_channels=depth_channels, with_depth=args.with_depth)
        ec = effnets.__efficient_channels__[args.model_version]
        encoder_channels = (3, ec[0], ec[1], ec[2], ec[3], ec[4])
    elif args.model_name == 'EfficientNet_x->xd':
        print('yes v2 here')
        encoder = effnets.EfficientNetRGBDv2(args.model_version, num_classes=out_dims, pretrained=pretrained,
                                             return_layers=return_layers,
                                             depth_channels=depth_channels, with_depth=args.with_depth)
        ec = effnets.__efficient_channels__[args.model_version]
        encoder_channels = (3, ec[0], ec[1], ec[2], ec[3], ec[4])
    elif args.model_name == 'Dino':
        if args.model_version == 'V1':
            encoder = get_dinov1_encoder(args)
            encoder_channels = [0] + [encoder.out_channels for _ in range(args.fpn_layers)]

        elif args.model_version == 'V2-small' or args.model_version == 'V2':
            encoder = get_dinov2_encoder(args, return_layers=return_layers, rgb_only=rgb_only,
                                         vit_return_layers=vit_return_layers,
                                         cat_outs=cat_outs, 
                                         resize_layers=resize_layers,
                                         down_sample_last_layer=down_sample_last_layer,
                                         norm_return_layers=norm_return_layers, 
                                         fusion_layer_ids=fusion_layer_ids,
                                         norm_return_layers_rgb=norm_return_layers_rgb,
                                         return_before_fusion=return_before_fusion,
                                         rand_disable_modality=rand_disable_modality,
                                         add_depth_scales_to_cls_token=add_depth_scales_to_cls_token,
                                         dino_size='small',
                                         fuse_cls_tolken=True)            
            encoder_channels = [0] + [encoder.out_channels for _ in range(args.fpn_layers)]
            decoder_channels = [256, 128, 64, 32, 16][:args.fpn_layers]
    
        elif args.model_version == 'V2-base':
            encoder = get_dinov2_encoder(args, return_layers=return_layers, rgb_only=rgb_only,
                                         vit_return_layers=vit_return_layers,
                                         cat_outs=cat_outs, resize_layers=resize_layers,
                                         down_sample_last_layer=down_sample_last_layer,
                                         norm_return_layers=norm_return_layers, fusion_layer_ids=fusion_layer_ids,
                                         norm_return_layers_rgb=norm_return_layers_rgb,
                                         return_before_fusion=return_before_fusion,
                                         rand_disable_modality=rand_disable_modality,
                                         add_depth_scales_to_cls_token=add_depth_scales_to_cls_token,
                                         dino_size='base')
            encoder_channels = [0] + [encoder.out_channels for _ in range(args.fpn_layers)]
        elif args.model_version == 'V2-large':
            encoder = get_dinov2_encoder(args, return_layers=return_layers, rgb_only=rgb_only,
                                         vit_return_layers=vit_return_layers,
                                         cat_outs=cat_outs, resize_layers=resize_layers,
                                         down_sample_last_layer=down_sample_last_layer,
                                         norm_return_layers=norm_return_layers, fusion_layer_ids=fusion_layer_ids,
                                         norm_return_layers_rgb=norm_return_layers_rgb,
                                         return_before_fusion=return_before_fusion,
                                         rand_disable_modality=rand_disable_modality,
                                         add_depth_scales_to_cls_token=add_depth_scales_to_cls_token,
                                         dino_size='large')
            encoder_channels = [0] + [encoder.out_channels for _ in range(args.fpn_layers)]
        elif args.model_version == 'V3-small':
            encoder = get_dinov3_encoder(args, return_layers=return_layers, rgb_only=rgb_only,
                                         vit_return_layers=vit_return_layers,
                                         cat_outs=cat_outs, resize_layers=resize_layers,
                                         down_sample_last_layer=down_sample_last_layer,
                                         norm_return_layers=norm_return_layers, fusion_layer_ids=fusion_layer_ids,
                                         norm_return_layers_rgb=norm_return_layers_rgb,
                                         return_before_fusion=return_before_fusion,
                                         rand_disable_modality=rand_disable_modality,
                                         add_depth_scales_to_cls_token=add_depth_scales_to_cls_token,
                                         dino_size='small')
            encoder_channels = [0] + [encoder.out_channels for _ in range(args.fpn_layers)]
        elif args.model_version == 'V3-base':
            encoder = get_dinov3_encoder(args, return_layers=return_layers, rgb_only=rgb_only,
                                         vit_return_layers=vit_return_layers,
                                         cat_outs=cat_outs, resize_layers=resize_layers,
                                         down_sample_last_layer=down_sample_last_layer,
                                         norm_return_layers=norm_return_layers, fusion_layer_ids=fusion_layer_ids,
                                         norm_return_layers_rgb=norm_return_layers_rgb,
                                         return_before_fusion=return_before_fusion,
                                         rand_disable_modality=rand_disable_modality,
                                         add_depth_scales_to_cls_token=add_depth_scales_to_cls_token,
                                         dino_size='base')
            encoder_channels = [0] + [encoder.out_channels for _ in range(args.fpn_layers)]
        elif args.model_version == 'V3-large':
            encoder = get_dinov3_encoder(args, return_layers=return_layers, rgb_only=rgb_only,
                                         vit_return_layers=vit_return_layers,
                                         cat_outs=cat_outs, resize_layers=resize_layers,
                                         down_sample_last_layer=down_sample_last_layer,
                                         norm_return_layers=norm_return_layers, fusion_layer_ids=fusion_layer_ids,
                                         norm_return_layers_rgb=norm_return_layers_rgb,
                                         return_before_fusion=return_before_fusion,
                                         rand_disable_modality=rand_disable_modality,
                                         add_depth_scales_to_cls_token=add_depth_scales_to_cls_token,
                                         dino_size='large')
            encoder_channels = [0] + [encoder.out_channels for _ in range(args.fpn_layers)]
        elif args.model_version == 'fit3D-small':
            encoder = get_fit3d_encoder(args, return_layers=return_layers, rgb_only=rgb_only,
                                         vit_return_layers=vit_return_layers,
                                         cat_outs=cat_outs, 
                                         resize_layers=resize_layers,
                                         down_sample_last_layer=down_sample_last_layer,
                                         norm_return_layers=norm_return_layers, 
                                         fusion_layer_ids=fusion_layer_ids,
                                         norm_return_layers_rgb=norm_return_layers_rgb,
                                         return_before_fusion=return_before_fusion,
                                         rand_disable_modality=rand_disable_modality,
                                         add_depth_scales_to_cls_token=add_depth_scales_to_cls_token,
                                         dino_size='small',
                                         fuse_cls_tolken=True) 
            encoder_channels = [0] + [encoder.out_channels for _ in range(args.fpn_layers)]
        elif args.model_version == 'fit3D-base':
            encoder = get_fit3d_encoder(args, return_layers=return_layers, rgb_only=rgb_only,
                                         vit_return_layers=vit_return_layers,
                                         cat_outs=cat_outs, 
                                         resize_layers=resize_layers,
                                         down_sample_last_layer=down_sample_last_layer,
                                         norm_return_layers=norm_return_layers, 
                                         fusion_layer_ids=fusion_layer_ids,
                                         norm_return_layers_rgb=norm_return_layers_rgb,
                                         return_before_fusion=return_before_fusion,
                                         rand_disable_modality=rand_disable_modality,
                                         add_depth_scales_to_cls_token=add_depth_scales_to_cls_token,
                                         dino_size='base',
                                         fuse_cls_tolken=True) 
            encoder_channels = [0] + [encoder.out_channels for _ in range(args.fpn_layers)]
        elif args.model_version == 'fit3D-large':
            encoder = get_fit3d_encoder(args, return_layers=return_layers, rgb_only=rgb_only,
                                         vit_return_layers=vit_return_layers,
                                         cat_outs=cat_outs, 
                                         resize_layers=resize_layers,
                                         down_sample_last_layer=down_sample_last_layer,
                                         norm_return_layers=norm_return_layers, 
                                         fusion_layer_ids=fusion_layer_ids,
                                         norm_return_layers_rgb=norm_return_layers_rgb,
                                         return_before_fusion=return_before_fusion,
                                         rand_disable_modality=rand_disable_modality,
                                         add_depth_scales_to_cls_token=add_depth_scales_to_cls_token,
                                         dino_size='large',
                                         fuse_cls_tolken=True) 
            encoder_channels = [0] + [encoder.out_channels for _ in range(args.fpn_layers)]
        
        elif args.model_version == 'dune-base':
            encoder = get_dune_encoder(args, return_layers=return_layers, rgb_only=rgb_only,
                                         vit_return_layers=vit_return_layers,
                                         cat_outs=cat_outs, 
                                         resize_layers=resize_layers,
                                         down_sample_last_layer=down_sample_last_layer,
                                         norm_return_layers=norm_return_layers, 
                                         fusion_layer_ids=fusion_layer_ids,
                                         norm_return_layers_rgb=norm_return_layers_rgb,
                                         return_before_fusion=return_before_fusion,
                                         rand_disable_modality=rand_disable_modality,
                                         add_depth_scales_to_cls_token=add_depth_scales_to_cls_token,
                                         dino_size='base',
                                         fuse_cls_tolken=True) 
            encoder_channels = [0] + [encoder.out_channels for _ in range(args.fpn_layers)]
        elif args.model_version == 'dune-small':
            encoder = get_dune_encoder(args, return_layers=return_layers, rgb_only=rgb_only,
                                         vit_return_layers=vit_return_layers,
                                         cat_outs=cat_outs, 
                                         resize_layers=resize_layers,
                                         down_sample_last_layer=down_sample_last_layer,
                                         norm_return_layers=norm_return_layers, 
                                         fusion_layer_ids=fusion_layer_ids,
                                         norm_return_layers_rgb=norm_return_layers_rgb,
                                         return_before_fusion=return_before_fusion,
                                         rand_disable_modality=rand_disable_modality,
                                         add_depth_scales_to_cls_token=add_depth_scales_to_cls_token,
                                         dino_size='small',
                                         fuse_cls_tolken=True) 
            encoder_channels = [0] + [encoder.out_channels for _ in range(args.fpn_layers)]

        else:
            raise NotImplementedError('The dino version "{}" is not Implemented'.format(args.model_version))
            
    elif args.model_name == 'OmniVore':
        if args.model_version in ['swinT', 'swinS', 'swinB', 'swinL']:
            encoder = OmniVoreWrapper(
                version=args.model_version, 
                resize_layers=resize_layers,
                down_sample_last_layer=down_sample_last_layer, 
                n_return_layers=4 if args.fpn_layers > 4 else args.fpn_layers, 
                num_classes=0, 
                freeze_encoder=freeze_color_encoder
                )

            encoder_channels = encoder.interm_channels
            if down_sample_last_layer:
                encoder_channels.append(encoder_channels-1)
            encoder_channels = [0] + [encoder_channels[i] for i in range(args.fpn_layers)]
        else:
            raise NotImplementedError('The OmniVore version "{}" is not Implemented'.format(args.model_version))
    elif args.model_name == 'DFormer':
        if args.model_version in ['tiny', 'small', 'base', 'large']:
            encoder = DformerWrapper(
                args.model_version,
                pretrained_path=args.pretrained_path,
                upsample_outs=upsample_outs
                )
            encoder_channels = [0] + encoder.interm_channels
            pretrained_loaded_in_get = True
        else:
            raise NotImplementedError('The DFormer version "{}" is not Implemented'.format(args.model_version))
    elif args.model_name == 'DFormerV2':
        if args.model_version in ['small', 'base', 'large']:
            encoder = DformerV2Wrapper(
                args.model_version,
                pretrained_path=args.pretrained_path,
                upsample_outs=upsample_outs
                )
            encoder_channels = [0] + encoder.interm_channels
            pretrained_loaded_in_get = True

        else:
            raise NotImplementedError('The DFormerV2 version "{}" is not Implemented'.format(args.model_version))
    elif args.model_name == 'MultiMAE':
        if args.model_version in ['base']:
            encoder = MultiMAEWrapper(upsample_outs=upsample_outs, rgb_only=rgb_only)
            encoder_channels = [0] + encoder.interm_channels
            #if args.with_depth and not rgb_only:
            #    for i in range(len(encoder_channels)):
            #        encoder_channels[i] = encoder_channels[i]*2
            pretrained_loaded_in_get = True

        else:
            raise NotImplementedError('The OmniVMultiMAEore version "{}" is not Implemented'.format(args.model_version))
    elif args.model_name == 'CroCoV2':
        if args.model_version in ['base']:
            encoder = CroCoEncoderWrapper(
                pretrained_path=args.pretrained_path, 
                upsample_outs=upsample_outs
            )
            encoder_channels = [0] + encoder.interm_channels
            pretrained_loaded_in_get = True
        else:
            raise NotImplementedError('The CroCoV2 version "{}" is not Implemented'.format(args.model_version))
    elif args.model_name == 'DepthAnythingV2':
        if args.model_version in ['base']:
            encoder = DepthAnythingV2Wrapper(
                upsample_outs=upsample_outs
            )
            encoder_channels = [0] + encoder.interm_channels
            pretrained_loaded_in_get = True
        else:
            raise NotImplementedError('The DepthAnythingV2 version "{}" is not Implemented'.format(args.model_version))
    elif args.model_name == 'DinoV2DepthAnythingV2':
        if args.model_version in ['base']:
            encoder = DinV2andDepthAnythingV2Wrapper(
                upsample_outs=upsample_outs
            )
            encoder_channels = [0] + encoder.interm_channels
            pretrained_loaded_in_get = True
        else:
            raise NotImplementedError('The DinoV2DepthAnythingV2 version "{}" is not Implemented'.format(args.model_version))
    elif args.model_name == 'MASt3R':
        if args.model_version in ['large']:
            encoder = MASt3REncoderWrapper(
                pretrained_path=args.pretrained_path, 
                upsample_outs=upsample_outs
            )
            encoder_channels = [0] + encoder.interm_channels
            pretrained_loaded_in_get = True
        else:
            raise NotImplementedError('The MASt3R version "{}" is not Implemented'.format(args.model_version))
    elif args.model_name == 'OmniDC':
        if args.model_version in ['base']:
            encoder = OmniDCWrapper(depth_module_input_size=args.train_size, downsample_feature_maps=downsample_feature_maps)
            encoder_channels = [0] + encoder.interm_channels
            pretrained_loaded_in_get = True
        else:
            raise NotImplementedError('The MASt3R version "{}" is not Implemented'.format(args.model_version))
    elif args.model_name == 'DMD3C':
        if args.model_version in ['base']:
            encoder = DMD3CWrapper(downsample_feature_maps=downsample_feature_maps)
            encoder_channels = [0] + encoder.interm_channels
            pretrained_loaded_in_get = True
        else:
            raise NotImplementedError('The MASt3R version "{}" is not Implemented'.format(args.model_version))
    elif args.model_name == 'SigLip2':
        if args.model_version == 'base':
            encoder = get_siglip2_encoder(args, return_layers=return_layers, rgb_only=rgb_only,
                                        vit_return_layers=vit_return_layers,
                                        cat_outs=cat_outs, 
                                        resize_layers=resize_layers,
                                        down_sample_last_layer=down_sample_last_layer,
                                        norm_return_layers=norm_return_layers, 
                                        fusion_layer_ids=fusion_layer_ids,
                                        norm_return_layers_rgb=norm_return_layers_rgb,
                                        return_before_fusion=return_before_fusion,
                                        rand_disable_modality=rand_disable_modality,
                                        add_depth_scales_to_cls_token=add_depth_scales_to_cls_token,
                                        model_size='base',
                                        fuse_cls_tolken=True)
            
                                        
        elif args.model_version == 'large':
            encoder = get_siglip2_encoder(args, return_layers=return_layers, rgb_only=rgb_only,
                                        vit_return_layers=vit_return_layers,
                                        cat_outs=cat_outs, 
                                        resize_layers=resize_layers,
                                        down_sample_last_layer=down_sample_last_layer,
                                        norm_return_layers=norm_return_layers, 
                                        fusion_layer_ids=fusion_layer_ids,
                                        norm_return_layers_rgb=norm_return_layers_rgb,
                                        return_before_fusion=return_before_fusion,
                                        rand_disable_modality=rand_disable_modality,
                                        add_depth_scales_to_cls_token=add_depth_scales_to_cls_token,
                                        model_size='large',
                                        fuse_cls_tolken=True) 
            
        else:
            raise NotImplementedError('The siglip2 version "{}" is not Implemented'.format(args.model_version))
        encoder_channels = [0] + [encoder.out_channels for _ in range(args.fpn_layers)]
        pretrained_loaded_in_get = True

    
    elif args.model_name == 'EVAv2':
        if args.model_version in ['base', 'small', 'large']:

            encoder = get_evav2_encoder(args, return_layers=True, vit_return_layers=vit_return_layers, rgb_only=rgb_only, cat_outs=cat_outs,
                       fusion_layer_ids=fusion_layer_ids, resize_layers=resize_layers, down_sample_last_layer=down_sample_last_layer,
                       return_before_fusion=None, rand_disable_modality=rand_disable_modality,
                       add_depth_scales_to_cls_token=add_depth_scales_to_cls_token, eva_size='base', img_size=args.train_size)        

            encoder_channels = [0] + [encoder.out_channels for _ in range(args.fpn_layers)]

        else:
            raise NotImplementedError('The OmniVore version "{}" is not Implemented'.format(args.model_version))
    
    else:
        raise NotImplementedError('Model {} not implemented'.format(args.model_name))

    if cp is not None and not pretrained_loaded_in_get:
        if 'state_dict' in cp:
            sd = list(encoder.state_dict().keys())
            del_keys = []
            for key in cp['state_dict']:
                if key not in sd:
                    del_keys.append(key)
            for key in del_keys:
                del cp['state_dict'][key]

            encoder.load_state_dict(cp['state_dict'], strict=False)
            print('Loaded encoder {}/{} weights for {} from {}'.format(len(cp['state_dict'].keys()),
                                                               len(cp['state_dict'].keys()) + len(del_keys),
                                                               len(sd), args.pretrained_path))
        elif args.model_name not in ['Dino', 'EVAv2']:
            encoder.color_encoder.arch = load_fitting_state_dict(encoder.color_encoder.arch, cp)
        

    if with_generator:
        if args.generate_depth_input:
            print('building generator')
            generator = build_generator(args.generator_path, args)
            encoder = (encoder, generator)

    try:
        if args.freeze_encoder:
            print('[info build encoder] frezeing encoder')
            for p in encoder.parameters():
                p.requires_grad = False    
    except Exception as e:
        print('[Warining build encoder]', e)


    return encoder, encoder_channels, decoder_channels


class DepthGenerator(nn.Module):
    def __init__(self, gen, dpe_in, dpe_out=None, const_dists=None, generator_batch_size=128):
        super(DepthGenerator, self).__init__()

        self.gen = gen
        for param in self.gen.parameters():
            param.requires_grad = False
        self.dpe_in = dpe_in
        self.dpe_out = dpe_out
        self.const_dists = const_dists
        self.generator_batch_size = generator_batch_size

    def forward(self, x, depth=None):
        self.gen.eval()
        bs, c, h, w = x.shape

        if not isinstance(depth, torch.Tensor):
            xd = torch.zeros((bs, 1, h, w), dtype=x.dtype, device=x.device)
            if depth is None:
                #print('depth is None')
                depth = self.const_dists

            if depth is not None:  # const_dists can be None as well
                for a, b, dist in depth:
                    if a < 1:
                        a = int(a * h)
                    if b < 1:
                        b = int(b * w)
                    #print('setting', a, b, dist)
                    xd[:, :, a, b] = dist
            #print(xd.shape, torch.max(xd))
            xd = self.dpe_in({'depth': xd})['depth']
            #print('xsd, ', xd.shape)
        else:
            xd = depth
            if len(xd.shape) > 4:
                bs_, i, c, h, w = xd.shape
                xd = xd.view(bs_*i, c, h, w)

        with torch.no_grad():
            #print('bs', bs, self.generator_batch_size)
            if bs > self.generator_batch_size:
                out = []
                for sel in torch.split(torch.arange(bs), self.generator_batch_size):
                    #print(len(sel), 'sel', sel)
                    pred = self.gen(x[sel], xd[sel])
                    if isinstance(pred, dict):
                        pred = pred['maps']
                    if isinstance(pred, list):
                        pred = pred[-1]
                    #print(torch.mean(pred), torch.std(pred), torch.min(pred), torch.max(pred))
                    #print('pred', pred.shape)
                    out.append(pred)
                    #print('outhere', out[-1].shape)
                #xd = out
                xd = torch.cat(out, dim=0)
                #print('xd', xd.shape)


            else:
                xd = self.gen(x, xd)
            # print('outshape', xd.shape)


        if isinstance(xd, dict):
            xd = xd['maps']
        if isinstance(xd, list):
            xd = xd[-1]

        #F.normalize(tensor, self.mean, self.std, self.inplace)
        #import matplotlib.pyplot as plt
        #from torchvision.transforms.functional import normalize
        #for x_, xd_ in zip(x, xd):
        #    x_ = normalize(x_,
        #                   [-0.485/0.229, -0.456/0.224, -0.406/0.225],
        #                   [1/0.229, 1/0.224, 1/0.225], False)
        #    img = x_.cpu().numpy().transpose(1, 2, 0)
        #    depth = xd_[0].cpu().numpy()
        #    plt.subplot(1, 2, 1)
        #    plt.imshow(img)
        #    plt.subplot(1, 2, 2)
        #    plt.imshow(depth)
        #    plt.show()

        '''
        import matplotlib.pyplot as plt
        im = x[0].cpu().numpy().transpose((1,2,0))
        print(im.shape)
        dists = xd[0, 0].cpu().numpy()

        plt.subplot(1,2,1)
        plt.imshow(im)
        plt.subplot(1, 2, 2)
        plt.imshow(dists)
        plt.show()
        '''

        if (h, w) != xd.shape[-2:]:
            xd = tvf.resize(xd, (h, w), InterpolationMode.NEAREST)

        xd = self.dpe_out({'depth': xd})['depth']

        return xd


def build_generator(path, args_encoder):
    cp = torch.load(path)
    from main import get_args_parser as main_args
    import argparse

    parser = argparse.ArgumentParser('Vanishing Depth training script', parents=[main_args()])
    args = parser.parse_args()
    #print('Loading args from pretrained checkpoint')
    for arg, v in cp['logs']['args'].items():
        #if v != getattr(args, arg):
        #    print('Setting arg {}: from {} -> {}'.format(arg, getattr(args, arg), v))
        setattr(args, arg, v)
        # print(arg, getattr(args, arg))

    generator = build_model(args)

    print('loading generator state dict')
    generator = load_fitting_state_dict(generator, cp['state_dict'])

    dpe_in = DepthPositionalEncoding(depth_channels=args.depth_channels,
                                     temperature=args.temperature,
                                     with_cos=args.with_cos,
                                     position_offset=args.position_offset,
                                     scale=args.scale)
    dpe_out = DepthPositionalEncoding(depth_channels=args_encoder.depth_channels,
                                      temperature=args_encoder.temperature,
                                      with_cos=args_encoder.with_cos,
                                      position_offset=args_encoder.position_offset,
                                      scale=args_encoder.scale)

    const_dist = None
    try:
        const_dist = args_encoder.const_dist
        if isinstance(const_dist, str):
            if const_dist == 'center 1':
                const_dist = [[0.5, 0.5, 1.0]]
            else:
                const_dist = None
    except:
        pass

    depth_generator = DepthGenerator(generator, dpe_in, dpe_out, const_dist,
                                     generator_batch_size=args_encoder.generator_batch_size)

    return depth_generator


class UNetDecoder(nn.Module):
    def __init__(self, encoder_channels, decoder_channels, n_blocks=4, patch_size=14):

        super(UNetDecoder, self).__init__()
        self.in_channels = encoder_channels[-1]
        self.decoder_channels = decoder_channels
        self.encoder_channels = encoder_channels
        self.decoder = d.UnetDecoder(encoder_channels[:n_blocks+1], decoder_channels[:n_blocks],
                                     fpn=False, n_blocks=n_blocks)
        self.patch_size = patch_size
        self.out_channels = decoder_channels[n_blocks-1]

    def forward(self, inputs, s):
        x = [x[:, 1:, :].permute(0, 2, 1).view(s[0], self.in_channels,
                                               s[2] // self.patch_size, s[3] // self.patch_size)
             for x in inputs]
        # s = x[0].shape
        # x = [x_[:, :, 1:, :].view(s[0], s[1], self.i_size[0], self.i_size[1]) for x_ in x]
        # x = [F.interpolate(x_, self.o_size) for i, x_ in enumerate(x)]
        out_shape = ((s[2] // 32) * 32, (s[3] // 32) * 32)

        x = [
            None,
            F.interpolate(x[3], (out_shape[0] // 2, out_shape[1] // 2)),
            F.interpolate(x[2], (out_shape[0] // 4, out_shape[1] // 4)),
            F.interpolate(x[1], (out_shape[0] // 8, out_shape[1] // 8)),
            F.interpolate(x[0], (out_shape[0] // 16, out_shape[1] // 16))
        ]
        #x = [x_[:, :, 1:, :].view(s[0], self.in_channels,
        #                          s[2] // self.patch_size, s[3] // self.patch_size) for x_ in x]
        return self.decoder(x)


def build_seg_model(args, vit_return_layers=None):
    if vit_return_layers is None:
        vit_return_layers = [2, 5, 8, 11]

    # print('decoder args before build encoder inside vanishing depth:',args.decoder)
    # input()
    encoder, encoder_channels, decoder_channels = build_encoder(args,
                                                                return_layers=True,
                                                                cat_outs=args.cat_outs,
                                                                resize_layers=False,
                                                                #vit_return_layers=[11] if args.ds_name == 'MVIP' else [8, 9, 10, 11],
                                                                vit_return_layers=vit_return_layers,
                                                                with_generator=True,
                                                                norm_return_layers=args.norm_return_layers,
                                                                norm_return_layers_rgb=args.norm_return_layers_rgb,
                                                                rand_disable_modality=args.rand_disable_modality)
    if isinstance(encoder, tuple):
        out_channels = encoder[0].out_channels
    else:
        out_channels = encoder.out_channels

    # print('DPT args inside Vanishing Depth is ', args.decoder)
    # input()
    if args.decoder == 'UNet':
        decoder = UNetDecoder(encoder_channels, (768, 384, 192, 96))

        head = SegmentationHead(
            in_channels=decoder.out_channels,
            out_channels=args.num_classes,
            activation=None,
            kernel_size=3)

    elif 'DPT' in args.decoder :
        from model.dpt import DPT, DPTSegHead, DPTDepthHead
        decoder = DPT(features=out_channels, multi_scale=args.multi_scale, num_register_tokens=encoder.num_register_tokens)
        print('decoder.out_channels', decoder.out_channels)
        print('multi_head', args.multi_head)
        print('multi_scale', args.multi_scale)
        if '-Depth' in args.decoder:
            print('DPTDepthHead')
            head = DPTDepthHead(features=decoder.out_channels, multi_head=args.multi_head,
                                decode_factors=args.decode_factors, activation=args.decode_activation, num_register_tokens=encoder.num_register_tokens)
        else:
            print('DPTSegHead')
            head = DPTSegHead(num_classes=args.num_classes, features=decoder.out_channels, multi_head=args.multi_head, num_register_tokens=encoder.num_register_tokens)
        input()
    elif args.decoder == 'linear':
        from model.linear_seghead import BNHead
        decoder = BNHead(in_channels=out_channels, layers=4, patch_size=14, out_channels=args.num_classes)
        head = nn.Identity()
    else:
        raise ValueError('Decoder {} not known'.format(args.decoder))

    return SegModel(encoder, decoder, head, args.num_classes, dropout=args.dropout, freeze_encoder=args.freeze_encoder)



if __name__ == '__main__':
    m = build_model(None)
    out = m(torch.rand(1, 3, 224, 224), torch.rand(1, 1, 224, 224))
    for o in out:
        print(o.shape)

