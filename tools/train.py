import os
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
os.environ['TORCH_USE_CUDA_DSA'] = "1"
os.environ['TORCH_DISTRIBUTED_DEBUG'] = "DETAIL"

import sys 
print('CWD', os.getcwd())
sys.path.append(os.getcwd())

import argparse
import logging

import os.path as osp

import torch
import torch.distributed as dist
#from torch.nn.parallel import DistributedDataParallel
from mmengine.config import Config, DictAction
from mmengine.logging import print_log
from mmengine.runner import Runner
from mmengine.hooks import Hook
from mmseg.registry import RUNNERS
import sys
sys.path.append('/home/chowanki/git/vanishing-depth-self-supervised')
from my_mmseg.datasets.CityscapesRGBD import *
from my_mmseg.datasets.Coco164kRGBD import *
from my_mmseg.datasets.ADE20kRGBD import *
from my_mmseg.datasets.SunRGBD import *
from my_mmseg.datasets.NYUDepthV2 import *
from my_mmseg.utils.data_preprocessor import *
from my_mmseg.model.vit14RGBD_Backbone import RGBDVit
print('imported my mmseg')
from mmseg.utils import register_all_modules
register_all_modules()
from my_mmseg.utils.transforms import *


def parse_args():
    parser = argparse.ArgumentParser(description='Train a segmentor')
    parser.add_argument('config', help='train config file path')
    parser.add_argument('--work-dir', help='the dir to save logs and models')
    parser.add_argument(
        '--resume',
        action='store_true',
        default=False,
        help='resume from the latest checkpoint in the work_dir automatically')
    parser.add_argument(
        '--amp',
        action='store_true',
        default=False,
        help='enable automatic-mixed-precision training')
    parser.add_argument(
        '--cfg-options',
        nargs='+',
        action=DictAction,
        help='override some settings in the used config, the key-value pair '
        'in xxx=yyy format will be merged into config file. If the value to '
        'be overwritten is a list, it should be like key="[a,b]" or key=a,b '
        'It also allows nested list/tuple values, e.g. key="[(a,b),(c,d)]" '
        'Note that the quotation marks are importnecessary and that no white space '
        'is allowed.')
    parser.add_argument(
        '--launcher',
        choices=['none', 'pytorch', 'slurm', 'mpi'],
        default='none',
        help='job launcher')
    # When using PyTorch version >= 2.0.0, the `torch.distributed.launch`
    # will pass the `--local-rank` parameter to `tools/train.py` instead
    # of `--local_rank`.
    parser.add_argument('--local_rank', '--local-rank', type=int, default=0)
    args = parser.parse_args()
    if 'LOCAL_RANK' not in os.environ:
        os.environ['LOCAL_RANK'] = str(args.local_rank)

    return args


class PrintDataHook(Hook):
    def __init__(self, log_file='data_batch_log.txt'):
        self.log_file = log_file
        # Initialize the log file by clearing its contents
        with open(self.log_file, 'w') as f:
            f.write("")

    def before_train_iter(self, runner, batch_idx, data_batch):
        with open(self.log_file, 'a') as f:
            # Print the type and content of the data batch
            f.write(f"Batch {batch_idx} - Data batch type: {type(data_batch)}\n")
            f.write(f"Batch {batch_idx} - Data batch content: {data_batch}\n")
            
            # Access and print the inputs and data samples correctly
            inputs = data_batch.get('inputs')
            data_samples = data_batch.get('data_samples')

            if inputs is not None:
                for i, tensor in enumerate(inputs):
                    f.write(f"Batch {batch_idx} - Input tensor {i} type: {type(tensor)}, shape: {tensor.shape}\n")
            if data_samples is not None:
                for i, sample in enumerate(data_samples):
                    f.write(f"Batch {batch_idx} - Data sample {i} type: {type(sample)}, content: {sample}\n")
                    gt_sem_seg = sample.gt_sem_seg.data
                    f.write(f"Batch {batch_idx} - Label tensor {i} type: {type(gt_sem_seg)}, shape: {gt_sem_seg.shape}\n")
                    f.write(f"Batch {batch_idx} - Label tensor {i} unique values: {gt_sem_seg.unique()}\n")

def main():
    args = parse_args()

    if args.launcher == 'pytorch':
        local_rank = int(os.environ['LOCAL_RANK'])
        torch.cuda.set_device(local_rank)
        dist.init_process_group(backend = 'nccl', init_method = 'env://')

    # load config
    cfg = Config.fromfile(args.config)
    cfg.launcher = args.launcher
    if args.cfg_options is not None:
        cfg.merge_from_dict(args.cfg_options)

    # print('Final Configuration:\n')
    # print(cfg.pretty_text)

    # input()

    # work_dir is determined in this priority: CLI > segment in file > filename
    if args.work_dir is not None:
        # update configs according to CLI args if args.work_dir is not None
        cfg.work_dir = args.work_dir
    elif cfg.get('work_dir', None) is None:
        # use config filename as default work_dir if cfg.work_dir is None
        cfg.work_dir = osp.join('./work_dirs',
                                osp.splitext(osp.basename(args.config))[0])

    # enable automatic-mixed-precision training
    if args.amp is True:
        optim_wrapper = cfg.optim_wrapper.type
        if optim_wrapper == 'AmpOptimWrapper':
            print_log(
                'AMP training is already enabled in your config.',
                logger='current',
                level=logging.WARNING)
        else:
            assert optim_wrapper == 'OptimWrapper', (
                '`--amp` is only supported when the optimizer wrapper type is '
                f'`OptimWrapper` but got {optim_wrapper}.')
            cfg.optim_wrapper.type = 'AmpOptimWrapper'
            cfg.optim_wrapper.loss_scale = 'dynamic'

    # resume training
    cfg.resume = args.resume

    cfg.model_wrapper_cfg = dict(
        type = 'MMDistributedDataParallel',
        find_unused_parameters= True,
        detect_anomalous_params = True
    )

    # build the runner from config
    if 'runner_type' not in cfg:
        # build the default runner
        runner = Runner.from_cfg(cfg)
    else:
        # build customized runner from the registry
        # if 'runner_type' is set in the cfg
        runner = RUNNERS.build(cfg)

    # Add debug prints here
    print("Starting training loop...")
    print("Batch size:", cfg.train_dataloader.batch_size)

    # Register print data hook
    runner.register_hook(PrintDataHook(), priority='HIGHEST')


    # start training
    runner.train()


if __name__ == '__main__':
    main()