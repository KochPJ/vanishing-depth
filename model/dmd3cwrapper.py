import os
import sys

if __name__ == '__main__':
    sys.path.append(os.getcwd())

import torch
import torch.nn as nn
import torch.nn.functional as F
from utils.stuff import load_fitting_state_dict

from model.DMD3C.BPNet import Net




class DMD3CWrapper(nn.Module):
    def __init__(self, rgb_only=False, downsample_feature_maps=False):
        super().__init__()

        self.arch = Net()
        sd = torch.load('./data/backbones/dmd3c_distillation_depth_anything_v2.pth')['net']
        #sd = torch.load('./data/backbones/dmd3c_pretrained_mixed_singleview_256.pth')['net']
        self.arch = load_fitting_state_dict(self.arch, sd)
        self.rgb_only = rgb_only
        self.downsample_feature_maps = downsample_feature_maps
        self.patch_size = 16
        if self.rgb_only:
            self.interm_channels = [64, 128, 256, 512]
        else:
            self.interm_channels = [128, 256, 512, 1024]
        
    def forward(self, x, xd, depth_scales=None):
        #print(x.shape, xd.shape)
        
        out = self.arch(x, xd)
        #print('out', [o.shape for o in out])
        
        if self.downsample_feature_maps:
            for i, o in enumerate(out):
                h, w = o.shape[-2:]
                out[i] = F.interpolate(o, (h//2, w//2), None, 'bilinear', None, recompute_scale_factor=None)            
                #print(i, o.shape, out[i].shape)

        return [None] + out + [None]


if __name__ == '__main__':

    device = torch.device('cuda:0')

    model = DMD3CWrapper().to(device)
    model.eval()

    x = torch.rand((1, 3, 480, 704), device=device)
    xd = torch.rand((1, 1, 480, 704), device=device)
    with torch.no_grad():
        out = model(x, xd)


    for i, o in enumerate(out):
        if o is None:
            print(i, type(o))
            
        else:
            print(i, type(o), o.shape)


'''
if __name__ == '__main__':
    from model.DMD3C.BPNet import Net
    from PIL import Image
    import numpy as np
    import matplotlib.pyplot as plt
    from torchvision import transforms


    device = torch.device('cuda:0')
    sd = torch.load('./data/backbones/dmd3c_distillation_depth_anything_v2.pth')['net']

    model = Net()
    model.load_state_dict(sd)

    model = model.to(device)

    model.eval()

    gt_ = np.array(Image.open('./data/kitti_depth_completion/depth_selection/val_selection_cropped/groundtruth_depth/2011_09_26_drive_0002_sync_groundtruth_depth_0000000005_image_02.png'), dtype=np.float32)/ 256
    sparse_depth_ = np.array(Image.open('./data/kitti_depth_completion/depth_selection/val_selection_cropped/velodyne_raw/2011_09_26_drive_0002_sync_velodyne_raw_0000000005_image_02.png'), dtype=np.float32)/ 256
    img_ = np.array(Image.open('./data/kitti_depth_completion/depth_selection/val_selection_cropped/image/2011_09_26_drive_0002_sync_image_0000000005_image_02.png'))
      

    sparse_depth = torch.as_tensor(sparse_depth_, dtype=torch.float32)
    gt = torch.as_tensor(gt_, dtype=torch.float32)
    img = torch.as_tensor(img_, dtype=torch.float32).permute((2,0,1))  / 255


    norm = transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    img = norm(img)
    K = torch.tensor([721.5377, 0.0, 596.5593, 0.0, 721.5377, 149.854, 0.0, 0.0, 1.0]).reshape((3,3)).unsqueeze(0)

    with torch.no_grad():
        out1 = model(
            I=img.to(device).unsqueeze(0), 
            S=sparse_depth.to(device).unsqueeze(0).unsqueeze(0), 
            #K=K.to(device)
            )[-1][0][0].detach().cpu().numpy()
        out2 = model(
            I=img.to(device).unsqueeze(0), 
            S=gt.to(device).unsqueeze(0).unsqueeze(0), 
            #K=K.to(device)
            )[-1][0][0].detach().cpu().numpy()
        
    plt.subplot(2, 3, 1)
    plt.imshow(img_)
    plt.title('RGB Image')
    plt.subplot(2, 3, 2)
    plt.title('Sparse Depth')
    plt.imshow(sparse_depth_)
    plt.subplot(2, 3, 3)
    plt.title('GT Depth')
    plt.imshow(gt_)
    plt.subplot(2, 3, 5)
    plt.title('Pred on Sparse Depth')
    plt.imshow(out1)
    plt.subplot(2, 3, 6)
    plt.title('Pred on GT Depth')
    plt.imshow(out2)
    plt.show()
'''