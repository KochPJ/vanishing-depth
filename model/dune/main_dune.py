import argparse
import datetime
import logging
import math
import os
import shutil
import time
import gc
from typing import Dict

import torch
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.nn as nn

from utils import exp
from utils.distributed import init_distributed_mode, get_global_rank
from utils.optim import get_params_groups, clip_gradients
from utils.vis import plot_arr
from data import get_dataset
from data.utils import my_collate
from data.sampler import InfiniteDistributedSampler
from model.dune import build_student_from_args
from model.teacher_dropping import TeacherDropping
from teachers import build_teachers


logger = logging.getLogger()


def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--enc_args",
        type=str,
        default="{}",
        help="Dictionary of keyword arguments for encoder. See model.options.EncoderOptions for all options.",
    )
    parser.add_argument(
        "--proj_args",
        type=str,
        default="{}",
        help="Dictionary of keyword arguments for projectors. See model.options.ProjectorOptions for all options.",
    )
    parser.add_argument(
        "--dune_args",
        type=str,
        default="{'last_enc_norm':False}",
        help="Dictionary of keyword arguments for DUNE.",
    )

    parser.add_argument(
        "--teachers",
        type=str,
        default="dino2reg_vitlarge_14,mast3r_vitlarge_16,multihmr_vitlarge_14_672",
        help="Comma-separated list of teacher names.",
    )
    parser.add_argument(
        "--tnorm_ema_momentum_start",
        type=float,
        default=1.0,
        help="Starting value for the EMA momentum for teacher feature statistics.",
    )
    parser.add_argument(
        "--tnorm_ema_momentum_end",
        type=float,
        default=0.001,
        help="Final value for the EMA momentum for teacher feature statistics.",
    )
    parser.add_argument(
        "--tdrop_args",
        type=str,
        default="{'method':'lowest_loss','p':0.5}",
        help="Arguments for the loss aggregator",
    )

    parser.add_argument(
        "--pretrained",
        type=str,
        default="",
        help="Path to the pretrained model checkpoint, to load just model weights.",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="teacher_balanced",
        help="Dataset name. See data/__init__.py for options.",
    )
    parser.add_argument(
        "--dataset_eval",
        type=str,
        default="in1k,Niantic,bedlam",
        help="Dataset name for evaluation. " "If not provided, --dataset will be used.",
    )
    parser.add_argument(
        "--image_size", type=int, default=336, help="Image size for training."
    )
    parser.add_argument(
        "--rrc_scale",
        type=str,
        default="0.40,1.0",
        help="Scale range for RandomResizedCrop augmentation. "
        "Two floating point values are expected, separated by comma.",
    )
    parser.add_argument(
        "--color_aug",
        type=exp.bool_flag,
        default=True,
        help="Whether to apply color augmentation to images or not.",
    )

    parser.add_argument(
        "--fp16",
        type=exp.bool_flag,
        default=True,
        help="Whether to use fp16 data types for forward passes",
    )
    parser.add_argument(
        "--compile",
        type=exp.bool_flag,
        default=False,
        help="Whether or not to compile the model.",
    )
    parser.add_argument(
        "--clip_grad",
        type=float,
        default=1.0,
        help="Gradient clipping value.",
    )
    parser.add_argument(
        "--batch_size_per_gpu",
        default=128,
        type=int,
        help="Batch size per GPU. Total batch size is proportional to the number of GPUs.",
    )
    parser.add_argument(
        "--epochs",
        default=100,
        type=float,
        help="Number of training epochs based on epoch_size.",
    )
    parser.add_argument(
        "--epoch_size",
        default=1281167,
        type=int,
        help="Number of samples in one epoch, by default the size of ImageNet-1K",
    )
    parser.add_argument(
        "--optim_args",
        type=str,
        default="{'betas':(0.9,0.99),'eps':1e-15,'fused':True}",
        help="Dictionary of keyword arguments for the optimizer.",
    )
    parser.add_argument(
        "--wd",
        type=float,
        default=3e-2,
        help="Weight decay for the SGD optimizer.",
    )
    parser.add_argument(
        "--lr",
        default=3e-4,
        type=float,
        help="Maximum learning rate at the end of linear warmup.",
    )
    parser.add_argument(
        "--min_lr",
        type=float,
        default=1e-6,
        help="Minimum learning rate at the end of training.",
    )
    parser.add_argument(
        "--warmup_epochs",
        default=-1,
        type=float,
        help="Number of training epochs for the learning-rate-warm-up phase. "
        "If negative, it is automatically set to '20%' of the total number of epochs.",
    )

    parser.add_argument(
        "--output_dir",
        default="./output",
        type=str,
        help="Path to the output folder to save logs and checkpoints.",
    )
    parser.add_argument(
        "--saveckpt_freq",
        default=20,
        type=int,
        help="Frequency of intermediate checkpointing, in terms of epochs.",
    )
    parser.add_argument(
        "--seed",
        default=22,
        type=int,
        help="Random seed",
    )
    parser.add_argument(
        "--num_workers",
        default=8,
        type=int,
        help="Number of data loading workers per GPU.",
    )
    parser.add_argument(
        "--dist_url",
        default="env://",
        type=str,
        help="Url used to set up distributed training.",
    )
    parser.add_argument(
        "--local_rank",
        default=0,
        type=int,
        help="Please ignore this argument; No need to set it manually.",
    )

    args = parser.parse_args()

    args.enc_args = eval(args.enc_args)
    args.proj_args = eval(args.proj_args)
    args.optim_args = eval(args.optim_args)
    args.tdrop_args = eval(args.tdrop_args)
    args.teachers = sorted(args.teachers.split(","))
    args.rrc_scale = list(map(float, args.rrc_scale.split(",")))
    args.num_cpus = len(os.sched_getaffinity(0))

    if args.warmup_epochs < 0:
        args.warmup_epochs = args.epochs * 0.2
        print("Warmup is automatically set to {}".format(args.warmup_epochs))

    os.makedirs(args.output_dir, exist_ok=True)

    return args


def main(args):
    init_distributed_mode(args)
    exp.fix_random_seeds(args.seed + get_global_rank())
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.set_printoptions(precision=4, sci_mode=False, linewidth=150, profile="short")
    cudnn.benchmark = True

    exp.configure_logger(
        output=os.path.join(args.output_dir, "log.txt"), level=logging.INFO
    )
    exp.print_program_info(args)
    ext_logger = exp.ExternalLogger(args.output_dir)

    logger.info("Loading teacher models ...")
    teachers = build_teachers(args.teachers)

    logger.info("Creating student model ...")
    model = build_student_from_args(args)
    logger.info("Trying to pretrained student from args.pretrained")
    exp.load_from_pretrained(model, args.pretrained, strict=True)
    model = model.cuda()

    if args.compile:
        logger.info("Compiling models ...")
        model = torch.compile(model)
        teachers = {k: torch.compile(v) for k, v in teachers.items()}

    model = nn.parallel.DistributedDataParallel(
        model, device_ids=[args.gpu], find_unused_parameters=True
    )
    exp.save_model_defn(model.module, os.path.join(args.output_dir, "model_defn.txt"))

    logger.info("Creating data loaders ...")
    train_loader, val_loader_list = get_dataloaders(args)

    optimizer = torch.optim.AdamW(
        get_params_groups(
            model,
            save_file_path=os.path.join(args.output_dir, "params_groups.txt"),
        ),
        lr=0,
        **args.optim_args,
    )
    logger.info("Optimizer: {}".format(optimizer))

    # Determine the number of iterations
    # and when to evaluate and save checkpoints
    in_debug_mode = os.environ.get("MYDEBUG", "False") == "True"
    args.batch_size = args.batch_size_per_gpu * dist.get_world_size()
    args.iters = math.ceil(args.epochs * args.epoch_size / (args.batch_size))
    args.iters = args.iters + (args.iters % 2)
    args.epoch_iters = (
        torch.linspace(
            0,
            args.iters,
            (
                # more frequent evaluations and checkpoints in debug mode
                (args.iters // 100) + 1
                if in_debug_mode
                # make at least args.saveckpt_freq evaluations
                else max(math.ceil(args.epochs), args.saveckpt_freq) + 1
            ),
        )
        .int()
        .tolist()[1:]
    )
    args.sep_ckpt_iters = args.epoch_iters[:: args.saveckpt_freq]
    logger.info("Total number of iterations: {:,}".format(args.iters))
    logger.info(
        "Model will be evaluated and saved "
        "at {} iterations: {}".format(len(args.epoch_iters), args.epoch_iters)
    )
    logger.info(
        "A separate checkpoint will be saved "
        "at {} iterations: {}".format(len(args.sep_ckpt_iters), args.sep_ckpt_iters)
    )
    assert (
        args.epoch_iters[-1] == args.iters
    ), "Last evaluation should be at the end of training"

    args.lr_schedule = exp.cosine_scheduler(
        args.lr * args.batch_size / 256.0,
        args.min_lr,
        epochs=1,  # args.epochs
        niter_per_ep=args.iters,  # len(train_loader),
        warmup_epochs=args.warmup_epochs / args.epochs,  # args.warmup_epochs,
    )
    plot_arr(args.lr_schedule, os.path.join(args.output_dir, "schedule_lr.png"))

    args.tnorm_ema_schedule = exp.cosine_scheduler(
        args.tnorm_ema_momentum_start,
        args.tnorm_ema_momentum_end,
        epochs=1,  # args.epochs
        niter_per_ep=args.iters,  # len(train_loader),
        warmup_epochs=0,
    )
    plot_arr(
        args.tnorm_ema_schedule, os.path.join(args.output_dir, "schedule_tnorm_ema.png")
    )

    logger.info("Trying to restart from previous checkpoint")
    to_restore = {"iter": 0}
    exp.restart_from_checkpoint(
        os.path.join(args.output_dir, "checkpoint.pth"),
        run_variables=to_restore,
        model=model,
        optimizer=optimizer,
    )
    start_iter = to_restore["iter"]

    logger.info(
        "Training starts at iteration: {:,}, remaining iterations: {:,}".format(
            start_iter, args.iters - start_iter
        )
    )
    start_time = time.time()

    training_loop(
        model,
        teachers,
        train_loader,
        val_loader_list,
        optimizer,
        ext_logger,
        args,
        start_iter,
    )

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    logger.info(
        "Training time {} for {:,} iterations".format(
            total_time_str, args.iters - start_iter
        )
    )


def training_loop(
    model,
    teachers,
    train_loader,
    val_loader_list,
    optimizer,
    ext_logger,
    args,
    start_iter,
):
    metrics_file = os.path.join(args.output_dir, "metrics_training.json")
    metric_logger = exp.MetricLogger(delimiter="  ", output_file=metrics_file)
    header = "Training"

    model.train()
    tdrop = get_tdrop(args.tdrop_args)

    for it, batch in enumerate(
        metric_logger.log_every(
            train_loader,
            20,
            header,
            n_iterations=args.iters,
            start_iteration=start_iter,
            ext_logger=ext_logger,
            ext_logger_prefix="train/",
            reset_after_print=True,
        )
    ):
        it = start_iter + it
        metric_dict = train_one_step(batch, model, teachers, tdrop, optimizer, args, it)
        metric_logger.update(**metric_dict)

        if (it + 1) in args.epoch_iters:
            run_evaluations(
                model,
                teachers,
                val_loader_list,
                it,
                optimizer,
                ext_logger,
                args,
            )
            model.train()


def train_one_step(batch, model, teachers, tdrop, optimizer, args, it):
    """
    One training iteration: forward, backward and optimizer step.
    """
    # Prepare the batch
    if args.dataset == "teacher_balanced":
        image, target, dset_name = prepare_teacher_balanced_batch(
            batch, args.batch_size_per_gpu
        )
    else:
        image, target, dset_name = batch

    image = image.cuda(non_blocking=True)

    for i, param_group in enumerate(optimizer.param_groups):
        param_group["lr"] = args.lr_schedule[it]
        if i == 0:
            param_group["weight_decay"] = args.wd

    metric_dict = {
        "lr": optimizer.param_groups[0]["lr"],
        "wd": optimizer.param_groups[0]["weight_decay"],
        "tnorm_ema_momentum": args.tnorm_ema_schedule[it],
    }

    with torch.autocast(enabled=args.fp16, device_type="cuda", dtype=torch.bfloat16):
        loss, loss_dict = model(
            image,
            dset_name,
            teachers,
            tdrop,
            args.tnorm_ema_schedule[it],
        )

    exp.check_loss(loss)
    loss.backward()

    metric_dict.update(loss_dict)

    if args.clip_grad > 0:
        grad_norms = clip_gradients(model, args.clip_grad)
        metric_dict.update(
            {
                "grad_norm/mean": grad_norms.mean().item(),
                "grad_norm/std": grad_norms.std().item(),
                "grad_norm/max": grad_norms.max().item(),
                "grad_norm/min": grad_norms.min().item(),
            }
        )

    optimizer.step()
    optimizer.zero_grad(set_to_none=True)

    return metric_dict


def run_evaluations(
    model: nn.Module,
    teachers: Dict[str, nn.Module],
    val_loader_list: list,
    it: int,
    optimizer: torch.optim.Optimizer,
    ext_logger,
    args,
):
    clear_cache()
    model.eval()

    logger.info("=> Evaluating at iteration {}".format(it))
    teval0 = time.time()

    for val_loader in val_loader_list:
        evaluate(
            model,
            teachers,
            val_loader,
            it,
            ext_logger,
            args,
        )
        clear_cache()

    teval1 = time.time()
    logger.info(
        "=> Evaluations finished in {}".format(
            str(datetime.timedelta(seconds=int(teval1 - teval0)))
        )
    )

    if get_global_rank() == 0:
        torch.save(
            {
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "iter": it + 1,
                "args": args,
            },
            os.path.join(args.output_dir, "checkpoint.pth"),
        )
        if (it + 1) in args.sep_ckpt_iters:
            shutil.copy(
                os.path.join(args.output_dir, "checkpoint.pth"),
                os.path.join(args.output_dir, f"checkpoint_{it:07}.pth"),
            )


@torch.inference_mode()
def evaluate(
    model,
    teachers,
    data_loader,
    curr_iter,
    ext_logger,
    args,
):
    metric_logger = exp.MetricLogger(delimiter="  ")
    dname_header = data_loader.dataset.dataset_name
    header = "Test {} - [{}/{}]".format(dname_header, curr_iter, args.iters)

    model.eval()
    tdrop = get_tdrop("")  # Predict all images by all teachers at test time

    n_iters = 100 if os.environ.get("MYDEBUG", "False") == "True" else len(data_loader)

    for it, batch in enumerate(
        metric_logger.log_every(data_loader, 10, header, n_iterations=n_iters)
    ):

        image, _, dset_name = batch
        image = image.cuda(non_blocking=True)

        with torch.autocast(
            enabled=args.fp16, device_type="cuda", dtype=torch.bfloat16
        ):
            _, loss_dict = model(
                image,
                dset_name,
                teachers,
                tdrop,
            )

        metric_logger.update(**loss_dict)

    metric_logger.synchronize_between_processes()
    metric_dict = {k: meter.global_avg for k, meter in metric_logger.meters.items()}

    logger.info("Averaged test stats:")
    for k, v in metric_dict.items():
        logger.info("{}: {}".format(k, v))

    ext_logger.log(
        metric_dict,
        curr_iter,
        prefix="test/{}/".format(dname_header),
        save_path=os.path.join(args.output_dir, "log_test.txt"),
    )

    return metric_dict


def prepare_teacher_balanced_batch(batch, batch_size_per_gpu):
    image = torch.cat([b[0] for b in batch])
    target = torch.cat([b[1] for b in batch])
    dset_name = sum([b[2] for b in batch], [])

    if len(image) > batch_size_per_gpu:
        image = image[:batch_size_per_gpu]
        target = target[:batch_size_per_gpu]
        dset_name = dset_name[:batch_size_per_gpu]

    return image, target, dset_name


def get_dataloaders(args):
    train_dataset = get_dataset(
        args.dataset,
        "train",
        image_size=args.image_size,
        rrc_scale=args.rrc_scale,
        color_aug=args.color_aug,
    )
    logger.info("Training dataset:\n - {}".format(train_dataset))

    # when dataset = teacher_balanced
    # dataset returns a sample for all teachers
    # to compensate, we modify the batch size for dataloader
    batch_size_per_gpu = args.batch_size_per_gpu
    if args.dataset == "teacher_balanced":
        batch_size_per_gpu = math.ceil(args.batch_size_per_gpu / 3)
        logger.info(
            "batch_size_per_gpu is set to {} for train split".format(batch_size_per_gpu)
        )

    train_loader = torch.utils.data.DataLoader(  # type: ignore
        train_dataset,
        sampler=InfiniteDistributedSampler(train_dataset, seed=args.seed, shuffle=True),
        batch_size=batch_size_per_gpu,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True,
        collate_fn=my_collate,
    )

    if args.dataset_eval == "":
        args.dataset_eval = args.dataset
    val_dataset_list = get_dataset(args.dataset_eval, "val", args.image_size)
    for val_dataset in val_dataset_list:
        logger.info("Validation dataset:\n - {}".format(val_dataset))

    val_loader_list = [
        torch.utils.data.DataLoader(
            val_dataset,
            batch_size=args.batch_size_per_gpu,
            num_workers=args.num_workers,
            shuffle=False,
            drop_last=False,
            pin_memory=True,
            collate_fn=my_collate,
        )
        for val_dataset in val_dataset_list
        if len(val_dataset) > 0
    ]

    return train_loader, val_loader_list


def get_tdrop(tdrop_args: Dict):
    if tdrop_args == "" or tdrop_args == "none":
        tdrop_args = {"method": "none", "p": 0}

    tdrop = TeacherDropping(**tdrop_args)
    return tdrop


def clear_cache():
    torch.cuda.empty_cache()
    gc.collect()


if __name__ == "__main__":
    args = get_args()
    main(args)
