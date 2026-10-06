# Copyright (C) 2022-present Naver Corporation. All rights reserved.
# Licensed under CC BY-NC-SA 4.0 (non-commercial use only).


# --------------------------------------------------------
# CroCo model during pretraining
# --------------------------------------------------------


import os
import sys
if __name__ == '__main__':
    sys.path.append(os.getcwd())

import torch
import torch.nn as nn

from functools import partial

from model.croco.models.blocks import Block, PatchEmbed
from model.croco.models.pos_embed import get_2d_sincos_pos_embed, RoPE2D 
from model.croco.models.masking import RandomMask
from utils.stuff import load_fitting_state_dict
import torch.nn.functional as F


class CroCoEncoderWrapper(nn.Module):

    def __init__(self,
                pretrained_path=None,
                upsample_outs=False,
                return_layers=[2, 5, 8, 11],
                img_size=224,           # input image size
                patch_size=16,          # patch_size 
                mask_ratio=0.9,         # ratios of masked tokens 
                enc_embed_dim=768,      # encoder feature dimension
                enc_depth=12,           # encoder depth 
                enc_num_heads=12,       # encoder number of heads in the transformer block 
                dec_embed_dim=768,      # decoder feature dimension 
                dec_depth=8,            # decoder depth 
                dec_num_heads=16,       # decoder number of heads in the transformer block 
                mlp_ratio=4,
                norm_layer=partial(nn.LayerNorm, eps=1e-6),
                norm_im2_in_dec=True,   # whether to apply normalization of the 'memory' = (second image) in the decoder 
                pos_embed='RoPE100',     # positional embedding (either cosine or RoPE100)
                ):

        torch.backends.cuda.matmul.allow_tf32 = True # for gpu >= Ampere and pytorch >= 1.12
                
        super(CroCoEncoderWrapper, self).__init__()
                
        # patch embeddings  (with initialization done as in MAE)
        self._set_patch_embed(img_size, patch_size, enc_embed_dim)

        # mask generations
        self._set_mask_generator(self.patch_embed.num_patches, mask_ratio)

        self.pos_embed = pos_embed
        if pos_embed=='cosine':
            # positional embedding of the encoder 
            enc_pos_embed = get_2d_sincos_pos_embed(enc_embed_dim, self.patch_embed.grid_size, n_cls_token=0)
            self.register_buffer('enc_pos_embed', torch.from_numpy(enc_pos_embed).float())
            # positional embedding of the decoder  
            dec_pos_embed = get_2d_sincos_pos_embed(dec_embed_dim, self.patch_embed.grid_size, n_cls_token=0)
            self.register_buffer('dec_pos_embed', torch.from_numpy(dec_pos_embed).float())
            # pos embedding in each block
            self.rope = None # nothing for cosine 
        elif pos_embed.startswith('RoPE'): # eg RoPE100 
            self.enc_pos_embed = None # nothing to add in the encoder with RoPE
            self.dec_pos_embed = None # nothing to add in the decoder with RoPE
            if RoPE2D is None: raise ImportError("Cannot find cuRoPE2D, please install it following the README instructions")
            freq = float(pos_embed[len('RoPE'):])
            self.rope = RoPE2D(freq=freq)
        else:
            raise NotImplementedError('Unknown pos_embed '+pos_embed)

        # transformer for the encoder 
        self.enc_depth = enc_depth
        self.enc_embed_dim = enc_embed_dim
        self.enc_blocks = nn.ModuleList([
            Block(enc_embed_dim, enc_num_heads, mlp_ratio, qkv_bias=True, norm_layer=norm_layer, rope=self.rope)
            for i in range(enc_depth)])
        self.enc_norm = norm_layer(enc_embed_dim)
        
        # masked tokens 
        self._set_mask_token(dec_embed_dim)

        # decoder 
        #self._set_decoder(enc_embed_dim, dec_embed_dim, dec_num_heads, dec_depth, mlp_ratio, norm_layer, norm_im2_in_dec)
        
        # prediction head 
        #self._set_prediction_head(dec_embed_dim, patch_size)
        
        # initializer weights
        self.initialize_weights()        

        pretrained_path = pretrained_path
        if pretrained_path:
            sd = torch.load(pretrained_path)

            print(sd.keys())
            print(sd['croco_kwargs'])
            sd = sd['model']
            
            # remove the decoder from the state dict
            for key in list(sd.keys()):
                if 'dec_' in key or 'decoder' in key or 'prediction_head' in key:
                    del sd[key]


            self = load_fitting_state_dict(self, sd)            

        self.upsample_outs = upsample_outs   
        self.return_layers = return_layers
        self.patch_size = patch_size
        self.interm_channels = [enc_embed_dim, enc_embed_dim, enc_embed_dim, enc_embed_dim]

    def _set_patch_embed(self, img_size=224, patch_size=16, enc_embed_dim=768):
        self.patch_embed = PatchEmbed(img_size, patch_size, 3, enc_embed_dim)

    def _set_mask_generator(self, num_patches, mask_ratio):
        self.mask_generator = RandomMask(num_patches, mask_ratio)
        
    def _set_mask_token(self, dec_embed_dim):
        self.mask_token = nn.Parameter(torch.zeros(1, 1, dec_embed_dim))
    
    '''
    def _set_decoder(self, enc_embed_dim, dec_embed_dim, dec_num_heads, dec_depth, mlp_ratio, norm_layer, norm_im2_in_dec):
        self.dec_depth = dec_depth
        self.dec_embed_dim = dec_embed_dim
        # transfer from encoder to decoder 
        self.decoder_embed = nn.Linear(enc_embed_dim, dec_embed_dim, bias=True)
        # transformer for the decoder 
        self.dec_blocks = nn.ModuleList([
            DecoderBlock(dec_embed_dim, dec_num_heads, mlp_ratio=mlp_ratio, qkv_bias=True, norm_layer=norm_layer, norm_mem=norm_im2_in_dec, rope=self.rope)
            for i in range(dec_depth)])
        # final norm layer 
        self.dec_norm = norm_layer(dec_embed_dim)
        
    def _set_prediction_head(self, dec_embed_dim, patch_size):
         self.prediction_head = nn.Linear(dec_embed_dim, patch_size**2 * 3, bias=True)
    '''
        
        
    def initialize_weights(self):
        # patch embed 
        self.patch_embed._init_weights()
        # mask tokens
        if self.mask_token is not None: torch.nn.init.normal_(self.mask_token, std=.02)
        # linears and layer norms
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            # we use xavier_uniform following official JAX ViT:
            torch.nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
            
    

    def unpatchify(self, x, channels=3):
        """
        x: (N, L, patch_size**2 *channels)
        imgs: (N, 3, H, W)
        """
        patch_size = self.patch_embed.patch_size[0]
        h = w = int(x.shape[1]**.5)
        assert h * w == x.shape[1]
        x = x.reshape(shape=(x.shape[0], h, w, patch_size, patch_size, channels))
        x = torch.einsum('nhwpqc->nchpwq', x)
        imgs = x.reshape(shape=(x.shape[0], channels, h * patch_size, h * patch_size))
        return imgs

    def forward(self, x, xd=None, depth_scales=None, mask=None):

        """
        image has B x 3 x img_size x img_size 
        do_mask: whether to perform masking or not
        return_all_blocks: if True, return the features at the end of every block 
                           instead of just the features from the last block (eg for some prediction heads)
        """
        #print('in', x.shape)
        H, W = x.shape[-2:]
        H_ = H // self.patch_size
        W_ = W // self.patch_size
        h_decode, w_decode = (H // 32)*32, (W // 32)*32
        # embed the image into patches  (x has size B x Npatches x C) 
        # and get position if each return patch (pos has size B x Npatches x 2)
        x, pos = self.patch_embed(x)              
        # add positional embedding without cls token  
        if self.enc_pos_embed is not None: 
            x = x + self.enc_pos_embed[None,...]
        # apply masking 
        B,N,C = x.size()
        
        B,N,C = x.size()
        masks = torch.zeros((B,N), dtype=bool)
        posvis = pos

        # now apply the transformer encoder and normalization        
        outs = []
        i = 0
        for ii, blk in enumerate(self.enc_blocks):
            x = blk(x, posvis)
            if ii in self.return_layers:
                s = self.enc_norm(x).view((B, H_, W_, C)).permute((0, 3, 1, 2))
                if self.upsample_outs:
                    t_size = (int(h_decode / (2**(i+1))), int(w_decode / (2**(i+1))))
                    s = F.interpolate(s, t_size, None, 'bilinear', None, recompute_scale_factor=None)

                outs.append(s)
                i+=1
        
        #print('out', [o.shape for o in outs])
        outs = [None] + outs + [None] # pad for consisten output (first = input size -> discarded anyways) last = cls token -> not needed and not used anyways

        return outs 


if __name__ == '__main__':

    device = torch.device('cuda:0')
    

    return_layers=[2, 5, 8, 11]
    model = CroCoEncoderWrapper(pretrained_path='./data/backbones/CroCo_V2_ViTBase_BaseDecoder.pth', upsample_outs=True).to(device)
    

    print(model)
    

    x = torch.rand((1, 3, 480, 688), device=device).to(device)
    xd = torch.rand((1, 1, 480, 688), device=device).to(device)

    out = model(x, xd)
    for i, o in enumerate(out):
        if o is None:
            print(i, type(o))
        else:
            print(i, type(o), o.shape)