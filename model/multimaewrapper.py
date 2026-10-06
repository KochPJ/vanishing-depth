import torch.nn as nn

import sys
import argparse
import os

try:

    import model.MultiMAE.utils as utils

    from model.MultiMAE.utils import data_constants
    import torch
    from model.MultiMAE.utils import create_model
    from functools import partial
    from model.MultiMAE.multimae import multimae
    from model.MultiMAE.multimae.criterion import (MaskedCrossEntropyLoss, MaskedL1Loss,
                                    MaskedMSELoss)
    from model.MultiMAE.multimae.input_adapters import PatchedInputAdapter, SemSegInputAdapter
    from model.MultiMAE.multimae.output_adapters import SpatialOutputAdapter
    from model.MultiMAE.utils import NativeScalerWithGradNormCount as NativeScaler
    from model.MultiMAE.utils import create_model
    from model.MultiMAE.utils.data_constants import COCO_SEMSEG_NUM_CLASSES
    from model.MultiMAE.utils.datasets import build_multimae_pretraining_dataset
    from model.MultiMAE.utils.optim_factory import create_optimizer
    from model.MultiMAE.utils.task_balancing import (NoWeightingStrategy,
                                    UncertaintyWeightingStrategy)

    from einops import rearrange, repeat
except:
    pass
import torch.nn.functional as F

class MultiMAEWrapper(nn.Module):
    
    def __init__(self, upsample_outs=False, rgb_only=False):
        super().__init__()
        self.upsample_outs = upsample_outs
                
        class Args:
            # Standardwerte wie im Parser
            batch_size = 256
            epochs = 1600
            save_ckpt_freq = 20
            in_domains = 'rgb-depth-semseg'
            out_domains = 'rgb-depth-semseg'
            standardize_depth = False
            extra_norm_pix_loss = True
            model = 'pretrain_multimae_base'
            num_encoded_tokens = 98
            num_global_tokens = 1
            patch_size = 16
            input_size = 224
            alphas = 1.0
            sample_tasks_uniformly = False
            drop_path_encoder = 0.0
            decoder_use_task_queries = True
            decoder_use_xattn = True
            decoder_dim = 256
            decoder_depth = 2
            decoder_num_heads = 8
            drop_path = 0.0
            loss_on_unmasked = False
            opt = 'adamw'
            opt_eps = 1e-8
            opt_betas = [0.9, 0.95]
            clip_grad = None
            skip_grad = None
            momentum = 0.9
            weight_decay = 0.05
            weight_decay_end = None
            decoder_decay = None
            blr = 1e-4
            warmup_lr = 1e-6
            min_lr = 0.0
            task_balancer = 'none'
            balancer_lr_scale = 1.0
            warmup_epochs = 40
            warmup_steps = -1
            fp32_output_adapters = ''
            hflip = 0.5
            train_interpolation = 'bicubic'
            data_path = 'DEIN/PFAD/ZU/DATEN'
            imagenet_default_mean_and_std = True
            output_dir = ''
            device = 'cuda'
            seed = 0
            resume = ''
            auto_resume = True
            start_epoch = 0
            num_workers = 10
            pin_mem = True
            find_unused_params = True
            log_wandb = False
            wandb_project = None
            wandb_entity = None
            wandb_run_name = None
            show_user_warnings = False
            world_size = 1
            local_rank = -1
            dist_on_itp = False
            dist_url = 'env://'

        args = Args()



        args.in_domains = args.in_domains.split('-')
        args.out_domains = args.out_domains.split('-')
        args.all_domains = list(set(args.in_domains) | set(args.out_domains))

                
        DOMAIN_CONF = {
            'rgb': {
                'channels': 3,
                'stride_level': 1,
                'input_adapter': partial(PatchedInputAdapter, num_channels=3),
                'output_adapter': partial(SpatialOutputAdapter, num_channels=3),
                'loss': MaskedMSELoss,
            },
            'depth': {
                'channels': 1,
                'stride_level': 1,
                'input_adapter': partial(PatchedInputAdapter, num_channels=1),
                'output_adapter': partial(SpatialOutputAdapter, num_channels=1),
                'loss': MaskedL1Loss,
            },
            'semseg': {
                'num_classes': 133,
                'stride_level': 4,
                'input_adapter': partial(SemSegInputAdapter, num_classes=COCO_SEMSEG_NUM_CLASSES,
                                        dim_class_emb=64, interpolate_class_emb=False),
                'output_adapter': partial(SpatialOutputAdapter, num_channels=COCO_SEMSEG_NUM_CLASSES),
                'loss': partial(MaskedCrossEntropyLoss, label_smoothing=0.0),
            },
        }
        

        input_adapters = {
            domain: DOMAIN_CONF[domain]['input_adapter'](
                stride_level=DOMAIN_CONF[domain]['stride_level'],
                patch_size_full=args.patch_size,
            )
            for domain in args.in_domains
        }

        output_adapters = {
            domain: DOMAIN_CONF[domain]['output_adapter'](
                stride_level=DOMAIN_CONF[domain]['stride_level'],
                patch_size_full=args.patch_size,
                dim_tokens=args.decoder_dim,
                depth=args.decoder_depth,
                num_heads=args.decoder_num_heads,
                use_task_queries=args.decoder_use_task_queries,
                task=domain,
                context_tasks=list(args.in_domains),
                use_xattn=args.decoder_use_xattn
            )
            for domain in args.out_domains
        }

        # Add normalized pixel output adapter if specified
        if args.extra_norm_pix_loss:
            output_adapters['norm_rgb'] = DOMAIN_CONF['rgb']['output_adapter'](
                stride_level=DOMAIN_CONF['rgb']['stride_level'],
                patch_size_full=args.patch_size,
                dim_tokens=args.decoder_dim,
                depth=args.decoder_depth,
                num_heads=args.decoder_num_heads,
                use_task_queries=args.decoder_use_task_queries,
                task='rgb',
                context_tasks=list(args.in_domains),
                use_xattn=args.decoder_use_xattn
            )

        self.model = create_model(
            args.model,
            input_adapters=input_adapters,
            output_adapters=output_adapters,
            drop_path_rate=args.drop_path_encoder,
        )

        sd = torch.load('./data/weights/multimae-b_98_rgb+-depth-semseg_1600e_multivit-afff3f8c.pth')['model']
        self.model.load_state_dict(sd)
        self.rgb_only = rgb_only
        if self.rgb_only:
            self.interm_channels = [768, 768, 768, 768]
        else:
            self.interm_channels = [768*2, 768*2, 768*2, 768*2]

        self.patch_size = (args.patch_size, args.patch_size) 


    def forward(self, x, xd=None, depth_scales=None, masks=None):

        #print(x.shape, xd.shape)        
        if xd is not None:
            x = {
                'rgb': x,
                'depth': xd
            }
        
        ## Processing input modalities
        # If input x is a Tensor, assume it's RGB
        x = {'rgb': x} if isinstance(x, torch.Tensor) else x

        # Need image size for tokens->image reconstruction
        # We assume that at least one of rgb or semseg is given as input before masking
        if 'rgb' in x:
            B, C, H, W = x['rgb'].shape
        elif 'semseg' in x:
            B, H, W = x['semseg'].shape
            H *= self.model.input_adapters['semseg'].stride_level
            W *= self.model.input_adapters['semseg'].stride_level
        else:
            B, C, H, W = list(x.values())[0].shape  # TODO: Deal with case where not all have same shape

        # Encode selected inputs to tokens
        input_task_tokens = {
            domain: self.model.input_adapters[domain](tensor)
            for domain, tensor in x.items()
            if domain in self.model.input_adapters
        }


        n_tasks = len(input_task_tokens)

        input_tokens = torch.cat([task_tokens for task_tokens in input_task_tokens.values()], dim=1)

        # Apply mask
        

        # Add global tokens to input tokens
        global_tokens = repeat(self.model.global_tokens, '() n d -> b n d', b=B)
        
        input_tokens = torch.cat([input_tokens, global_tokens], dim=1)
        #print('input_tokens', input_tokens.shape)

        ## Transformer forward pass
        encoder_tokens = input_tokens 
        maps = []
        fc = []
        i = 0
        for ii, enc in enumerate(self.model.encoder):
            encoder_tokens = enc(encoder_tokens)
            if ii in [2, 5, 8, 11]:
                
                fc.append(encoder_tokens[:, -1])
                s = encoder_tokens[:, :-1]
                ch = s.shape[-1]
                k = s.shape[1] // n_tasks
                s = s.split(k, dim=1)
                s = torch.cat([s_.view(B, H//16, W//16, ch).permute(0, 3, 1, 2) for s_ in s], dim=1)
                #print(ii, s.shape)

                
                if self.upsample_outs:
                    t_size = (int(H / (2**(i+1))), int(W / (2**(i+1))))
                    s = F.interpolate(s, t_size, None, 'bilinear', None, recompute_scale_factor=None)
                    #print('upsample', t_size, s.shape)

                maps.append(s)
                i += 1

        #encoder_tokens = self.encoder(input_tokens)
        maps = [None] + maps + [None] # pad for consisten output (first = input size -> discarded anyways) last = cls token -> not needed and not used anyways
        return maps
        #return (maps, fc)



if __name__ == '__main__':
    device = torch.device('cuda:0')
    model = MultiMAEWrapper(upsample_outs=True).to(device)


    x = torch.rand((1, 3, 224, 224)).to(device)
    xd = torch.rand((1, 1, 224, 224)).to(device)

    out = model(x, xd)
    print(type(out), len(out))
    for i, o in enumerate(out):
        if o is None:
            print(i, o)
            
        else:
            print(i, o.shape)

    #print(out)