import sys
import os
import pathlib
from timm.models import create_model
import timm
import torch

sys.path.append('/home/kochpaul/git/vanishing-depth-self-supervised')


try:
    from model.EVA.EVA2.asuka.eva_clip.vit_model import VisionTransformer
    from model.EVA.eva2 import EVA2
except Exception as e:
    #raise e
    VisionTransformer, EVA2 = None, None
from model.dino import DinoRGBDWrapper, DinoRGBWrapper
from utils.stuff import load_fitting_state_dict


abs_file_path = pathlib.Path(__file__).parent.resolve()#
sys.path.append(abs_file_path)



def evav2_base(patch_size=14, in_chans=3, return_layers=None, return_rdps=False, fusion_layer_ids=None, add_fusion=False, img_size=518, 
                return_before_fusion=False, add_depth_scales_to_cls_token=True, fuse_cls_tolken=False, num_classes=0, pretrained=None):

    if return_layers is None: 
        return_layers = [3, 5, 7, 11]

    
    if fusion_layer_ids is None: 
        fusion_layer_ids = [3, 5, 7, 11]

    

    eva = EVA2(
        
        img_size=img_size, 
        patch_size=patch_size, 
        in_chans=in_chans, 
        num_classes=num_classes, 
        embed_dim=768, 
        depth=12,
        num_heads=12, 
        mlp_ratio=4*2/3,      # GLU default
        
        qkv_bias=True, 
        qk_scale=None, 
        drop_rate=0., 
        attn_drop_rate=0.,
        drop_path_rate=0.15, 
        
        hybrid_backbone=None, 
        norm_layer=None, 
        
        init_values=None, 
        use_checkpoint=False, 

        use_abs_pos_emb=True, 
        use_rel_pos_bias=False, 
        use_shared_rel_pos_bias=False,
        out_indices=[3, 5, 7, 11],

        subln=True,
        xattn=True,
        naiveswiglu=True,
        rope=True,
        pt_hw_seq_len=16,
        intp_freq=True,

        pretrained=pretrained,

        # Vanishing Depth stuff
        add_fusion=add_fusion,
        return_layers=return_layers,
        return_rdps=return_rdps,
        fusion_layer_ids=fusion_layer_ids,
        return_before_fusion=return_before_fusion,
        add_depth_scales_to_cls_token=add_depth_scales_to_cls_token,
        fuse_cls_tolken=fuse_cls_tolken
    )

    return eva



def get_evav2_encoder(args, return_layers=True, vit_return_layers=None, rgb_only=False, cat_outs=False,
                    fusion_layer_ids=None, resize_layers=True, down_sample_last_layer=True,
                    return_before_fusion=None, rand_disable_modality=False,
                    add_depth_scales_to_cls_token=False, eva_size='base', fuse_cls_tolken=False, img_size=224,
                    patch_size = 14):

    eva_encoder = {
        'small': evav2_base, 
        'base': evav2_base,
        'large': evav2_base
    }[eva_size]


    if fusion_layer_ids is None:
        fusion_layer_ids = {
            'small': [2, 5, 8, 11],
            'base': [2, 5, 8, 11],
            'large': [5, 12, 18, 24]
        }[eva_size]

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
    print('rand_disable_modality', rand_disable_modality)
    print('add_depth_scales_to_cls_token', add_depth_scales_to_cls_token)
    print('eva_encoder', eva_encoder)
    print('fuse_cls_tolken', fuse_cls_tolken)

    if isinstance(args.pretrained_path, str):
        if 'p14to16' in args.pretrained_path:
            patch_size = 16
            img_size = 512
        if len(args.pretrained_path) == 0:
            args.pretrained_path = None


    print('patch_size', patch_size)
    print('img_size', img_size)
    print('pretrained', args.pretrained_path)


    vit_rgb = eva_encoder(patch_size=patch_size,
                           return_layers=vit_return_layers,
                           fusion_layer_ids=fusion_layer_ids,
                           img_size=img_size,
                           pretrained=args.pretrained_path)
                           
    print('return_rdps', args.return_rdps)

    if not rgb_only:
        depth_channels = args.depth_channels
        try:
            if args.with_intr:
                depth_channels = depth_channels * 3
        except:
            pass
        vit_depth = eva_encoder(patch_size=patch_size,
                                 in_chans=depth_channels,
                                 return_layers=vit_return_layers,
                                 return_rdps=args.return_rdps,
                                 fusion_layer_ids=fusion_layer_ids,
                                 add_fusion=True, img_size=img_size,
                                 return_before_fusion=return_before_fusion,
                                 add_depth_scales_to_cls_token=add_depth_scales_to_cls_token,
                                 fuse_cls_tolken=fuse_cls_tolken,
                                pretrained=args.pretrained_path)
    else:
        print('RGB Only encoder')

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

if __name__ == "__main__":

    from main import get_args_parser, main
    import argparse

    parser = argparse.ArgumentParser('Vanishing Depth training script', parents=[get_args_parser()])
    args = parser.parse_args()
    #args.pretrained_path = './data/backbones/eva02_B_pt_in21k_p14.pt'
    args.pretrained_path = './data/backbones/eva02_B_pt_in21k_p14to16.pt'


    model = get_evav2_encoder(args)
    device = torch.device('cpu')

    input()
    print(model)


    input()
    model = model.to(device)

    x = torch.rand((1, 3, 224, 224)).to(device)

    out = model(x)
    print(type(out))
    
    