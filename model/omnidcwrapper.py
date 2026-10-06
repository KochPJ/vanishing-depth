import os
import sys

if __name__ == '__main__':
    sys.path.append(os.getcwd())

import torch
import torch.nn as nn
import torch.nn.functional as F
from model.OmniDC.model.backbone import Backbone
from utils.stuff import load_fitting_state_dict

from model.OmniDC.model.optim_layer.optim_layer import DepthGradOptimLayer
from model.depth_anything_v2.depth_anything_v2.dpt import DepthAnythingV2
from model.OmniDC.model.align_utils import resize_image, depth2disparity, disparity2depth, align_least_square, align_single_res

dav2_model_configs = {
    'vits': {'encoder': 'vits', 'features': 64, 'out_channels': [48, 96, 192, 384]},
    'vitb': {'encoder': 'vitb', 'features': 128, 'out_channels': [96, 192, 384, 768]},
    'vitl': {'encoder': 'vitl', 'features': 256, 'out_channels': [256, 512, 1024, 1024]},
    'vitg': {'encoder': 'vitg', 'features': 384, 'out_channels': [1536, 1536, 1536, 1536]}
}



class OmniDCWrapper(nn.Module):
    def __init__(self, depth_module_input_size=518, downsample_feature_maps=False):
        super().__init__()
        
        sd = torch.load('./data/backbones/OmniDC_modelv1.1_best_72epochs.pt', map_location='cpu')
        self.args = sd['args']
        if sd['args'].load_dav2:
            depth_input_channels = 2
        else:
            depth_input_channels = 1


        print('backbone_mode', sd['args'].backbone_mode, depth_input_channels)
        print('backbone_pattern_condition_format', self.args.backbone_pattern_condition_format)
        print('whiten_sparse_depths', self.args.whiten_sparse_depths)

        self.downsample_rate = self.args.backbone_output_downsample_rate

        self.arch = Backbone(sd['args'], mode=sd['args'].backbone_mode, depth_input_channels=depth_input_channels)


        sd_ = {}
        for key, v in sd['net'].items():
            if 'backbone.' in key:
                sd_[key.replace('backbone.', '')] = v
        
        
        self.hdim = self.args.gru_hidden_dim
        self.cdim = self.args.gru_context_dim

        self.resolution = self.args.num_resolution

        encoder = 'vitl'        
        self.depth_module = DepthAnythingV2(**dav2_model_configs[encoder])
        self.depth_module.load_state_dict(
            torch.load(f'./data/backbones/depth_anything_v2_{encoder}.pth',
                        map_location='cpu'))
        self.depth_module = self.depth_module.eval()

        # freeze foundation model
        for param in self.depth_module.parameters():
            param.requires_grad = False

        self.patch_size = 16
        if depth_module_input_size%14 != 0:
            depth_module_input_size = 14 * (depth_module_input_size//14 + 1)    
        self.depth_module_input_size = depth_module_input_size
        self.downsample_feature_maps = downsample_feature_maps

        # NLSPN
        self.prop_time = self.args.prop_time
        if self.args.spn_type == "nlspn":
            from model.OmniDC.model.nlspn_module import NLSPN

            self.num_neighbors = args.prop_kernel * self.args.prop_kernel - 1
            if self.prop_time > 0:
                self.prop_layer = NLSPN(self.args, self.num_neighbors, 1, 3,
                                        self.args.prop_kernel)
        elif self.args.spn_type == "dyspn":
            from model.OmniDC.model.dyspn_module import DySPN_Module

            self.num_neighbors = 5
            if self.prop_time > 0:
                assert self.prop_time == 6
                self.prop_layer = DySPN_Module(iteration=self.prop_time,
                                               num=self.num_neighbors,
                                               mode='yx')
        else:
            raise NotImplementedError

        
        self.arch = load_fitting_state_dict(self.arch, sd_)

        self.interm_channels = [192, 128, 256, 576]
        self.arch.eval()
   
        
    def forward(self, rgb, dep_original, depth_scales=None):
        
        #rgb = x
        #dep_original = xd
        if dep_original.ndim == 3:
            dep_original = dep_original.unsqueeze(1)
        depth_pattern = None

        B, _, H, W = rgb.shape

        #print('rgb.shape', rgb.shape, dep_original.shape, rgb.device, dep_original.device)

        valid_sparse_mask = (dep_original > 0.0).float()
        #valid_sparse_mask_network_input = torch.clone(valid_sparse_mask)

        # this is full-res depth
        if self.args.whiten_sparse_depths:
            medians = torch.ones(B, device=rgb.device)
            for b in range(B):
                nonzeros = dep_original[b] > 0.0
                if len(nonzeros) > 0:
                    medians[b] = torch.median(dep_original[b][nonzeros])

            dep_network_input = dep_original / medians.reshape(B, 1, 1, 1)  # make the median to be always 1.0
        else:
            dep_network_input = torch.clone(dep_original)

        # sparse depth needs downsample before feeding into the optim layer
        #if self.downsample_rate > 1:
        #    
        #    if self.args.depth_downsample_method == "min":
        #        dep[dep == 0.0] = 100000.0  # set the invalid values to inf
        #        dep = -F.max_pool2d(-dep, self.downsample_rate)  # trick to do min-pooling
        #        valid_sparse_mask = F.max_pool2d(valid_sparse_mask,
        #                                         self.downsample_rate)  # mask is 1 if at least one pt in neighbor
        #        dep[valid_sparse_mask == 0.0] = 0.0  # set invalid value back to 0.0, for safety
        #    else:
        #        raise NotImplementedError

        if self.args.depth_activation_format == "exp":
            dep_network_input = torch.log(dep_network_input)
        else:
            dep_network_input = dep_network_input

        dep_network_input[valid_sparse_mask == 0.0] = 0.0

        if self.args.training_depth_random_shift_range > 0.0 and self.training:
            batch_size = rgb.shape[0]
            random_shift = torch.empty(batch_size).uniform_(-0.5,
                                                            0.5).to(dep_network_input.device) * self.args.training_depth_random_shift_range
            dep_network_input = dep_network_input + random_shift.reshape(batch_size, 1, 1, 1)

        if self.args.load_dav2:
            rgb_resized = resize_image(rgb, size=self.depth_module_input_size)  # B x 3 x 518 x W_resized

            # relative depth
            depth_pred_raw = self.depth_module.forward(rgb_resized).unsqueeze(1)  # B x 1 x 518 x W_resized
            depth_pred_raw = F.relu(depth_pred_raw)

            # resize back
            depth_pred_raw = F.interpolate(depth_pred_raw, (H, W), mode="bilinear",
                                            align_corners=True)  # B x 1 x H x W

            # normalize to [0,1]
            _min = torch.quantile(depth_pred_raw.reshape(B, -1), 0.02, dim=1).reshape(B, 1, 1, 1)
            _max = torch.quantile(depth_pred_raw.reshape(B, -1), 0.98, dim=1).reshape(B, 1, 1, 1)

            dav2_depth = 1.0 * (depth_pred_raw - _min) / (_max - _min)

            dep_network_input = torch.cat([dep_network_input, dav2_depth], dim=1)
        else:
            dep_network_input = dep_network_input

        # backbone
        assert self.args.pred_context_feature
        
        with torch.no_grad():
            out = self.arch(rgb,
                            dep_network_input,
                            depth_pattern,
                            return_encoding=True)

        if self.downsample_feature_maps:
            for i, o in enumerate(out):
                _, _, h, w = o.shape
                out[i] = F.interpolate(o, (h//2, w//2), None, 'bilinear', None, recompute_scale_factor=None)
                #print(o.shape, out[i].shape)

        #print('out', [o.shape for o in out])

        return [None] + out + [None]


if __name__ == '__main__':

    device = torch.device('cuda:0')

    model = OmniDCWrapper().to(device)
    model.eval()

    x = torch.rand((1, 3, 512, 1568), device=device)
    xd = torch.rand((1, 1, 512, 1568), device=device)
    with torch.no_grad():
        out = model(x, xd)
    for i, o in enumerate(out):
        if o is None:
            print(i, type(o))
            
        else:
            print(i, type(o), o.shape)

'''

if __name__ == '__main__':

    device = torch.device('cuda:0')

    sd = torch.load('./data/backbones/OmniDC_modelv1.1_best_72epochs.pt', map_location='cpu')
    args = sd['args']
    from model.OmniDC.model.ognidc import OGNIDC
    model = OGNIDC(args)

    model = load_fitting_state_dict(model, sd['net'])

    model = model.to(device)
    model.eval()

    print(model)
    from PIL import Image
    import numpy as np

    import matplotlib.pyplot as plt
    from torchvision import transforms

    gt_ = np.array(Image.open('./data/void_release/void_1500/data/classroom0/ground_truth/1552098057.8059.png'), dtype=np.float32)
    sparse_depth_ = np.array(Image.open('./data/void_release/void_1500/data/classroom0/sparse_depth/1552098057.8059.png'), dtype=np.float32)
    img_ = np.array(Image.open('./data/void_release/void_1500/data/classroom0/image/1552098057.8059.png'))

    #gt_ = np.array(Image.open('./data/kitti_depth_completion/depth_selection/val_selection_cropped/groundtruth_depth/2011_09_26_drive_0002_sync_groundtruth_depth_0000000005_image_02.png'), dtype=np.float32)/ 256
    #sparse_depth_ = np.array(Image.open('./data/kitti_depth_completion/depth_selection/val_selection_cropped/velodyne_raw/2011_09_26_drive_0002_sync_velodyne_raw_0000000005_image_02.png'), dtype=np.float32)/ 256
    #img_ = np.array(Image.open('./data/kitti_depth_completion/depth_selection/val_selection_cropped/image/2011_09_26_drive_0002_sync_image_0000000005_image_02.png'))

    


    sparse_depth = torch.as_tensor(sparse_depth_, dtype=torch.float32) / 1000
    gt = torch.as_tensor(gt_, dtype=torch.float32) / 1000
    img = torch.as_tensor(img_, dtype=torch.float32).permute((2,0,1))  / 255

    print('img', img.shape, torch.mean(img))

    norm = transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    img = norm(img)
    
    print('img', img.shape, torch.mean(img))
    print('gt', gt.shape, torch.mean(gt))
    print('sparse_depth', sparse_depth.shape, torch.mean(sparse_depth))

    K = torch.eye(3).reshape(1, 3, 3).cuda()

    sample = {
        'rgb': img.unsqueeze(0).to(device),
        'dep': sparse_depth.to(device).unsqueeze(0).unsqueeze(0),
        'K': K, # dummy one; not actually used
        'pattern': 0 # dummy one; not actually used
    }
    sample2 = {
        'rgb': img.unsqueeze(0).to(device),
        'dep': gt.to(device).unsqueeze(0).unsqueeze(0),
        'K': K, # dummy one; not actually used
        'pattern': 0 # dummy one; not actually used
    }

    with torch.no_grad():

        out1 = model(sample)['pred'][0][0].detach().cpu().numpy()
        out2 = model(sample2)['pred'][0][0].detach().cpu().numpy()
    
    plt.subplot(2, 3, 1)
    plt.imshow(img_)
    plt.subplot(2, 3, 2)
    plt.imshow(sparse_depth_)
    plt.subplot(2, 3, 3)
    plt.imshow(gt_)
    plt.subplot(2, 3, 5)
    plt.imshow(out1)
    plt.subplot(2, 3, 6)
    plt.imshow(out2)
    plt.show()



'''
