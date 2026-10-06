import os
import sys
import functools
import json
import math
import random
import argparse
import time
import logging
import datetime
from collections import defaultdict
from typing import Optional

import numpy as np
import torch
import torch.distributed as dist
from torch.utils.tensorboard import SummaryWriter

from .distributed import is_main_process, get_global_rank
from .metrics import SmoothedValue


logger = logging.getLogger()


def bool_flag(s):
    FALSY_STRINGS = {"off", "false", "0"}
    TRUTHY_STRINGS = {"on", "true", "1"}
    if s.lower() in FALSY_STRINGS:
        return False
    elif s.lower() in TRUTHY_STRINGS:
        return True
    else:
        raise argparse.ArgumentTypeError("Invalid value for a boolean flag")


def fix_random_seeds(seed=22):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def print_program_info(args):
    logger.info("Args:")
    for k, v in sorted(dict(vars(args)).items()):
        logger.info("\t{}: {}".format(k, str(v)))

    with open(os.path.join(args.output_dir, "args.json"), "w") as fp:
        json.dump(
            dict(vars(args)),
            fp,
            indent=4,
            sort_keys=True,
        )

    logger.info("Env vars:")
    for env_var in [
        "ONEDAL_NUM_THREADS",
        "MKL_NUM_THREADS",
        "OMP_NUM_THREADS",
        "KMP_AFFINITY",
        "KMP_BLOCKTIME",
        "MYDEBUG",
    ]:
        logger.info("\t{}={}".format(env_var, os.environ.get(env_var, "(unset)")))

    logger.info("Script caller: {}".format(sys.argv[0]))
    for parg in sys.argv[1:]:
        logger.info("\t{}".format(parg))


def save_model_defn(model, save_path):
    fp = open(os.path.join(save_path), "w")
    fp.write("{}".format(model))
    fp.write("\n")

    modules = {
        "model": model,
        "encoder": model.encoder,
        "projectors": model.projectors,
        "teacher_norms": model.teacher_norms,
    }

    for mname, module in modules.items():
        trainable = sum(p.numel() for p in module.parameters() if p.requires_grad)
        frozen = sum(p.numel() for p in module.parameters() if not p.requires_grad)
        fp.write(
            "Number of trainable parameters in {} : {:,}\n".format(mname, trainable)
        )
        fp.write("Number of frozen parameters in {} : {:,}\n".format(mname, frozen))

    fp.flush()
    fp.close()


def cosine_scheduler(
    base_value, final_value, epochs, niter_per_ep, warmup_epochs=0, start_warmup_value=0
):
    """
    Creates a cosine scheduler with linear warm-up.
    """
    warmup_schedule = np.array([])
    warmup_iters = math.floor(warmup_epochs * niter_per_ep)
    if warmup_epochs > 0:
        warmup_schedule = np.linspace(start_warmup_value, base_value, warmup_iters)

    iters = np.arange(math.floor(epochs * niter_per_ep) - warmup_iters)
    schedule = final_value + 0.5 * (base_value - final_value) * (
        1 + np.cos(np.pi * iters / len(iters))
    )

    schedule = np.concatenate((warmup_schedule, schedule))
    assert len(schedule) == math.floor(epochs * niter_per_ep)
    return schedule


def restart_from_checkpoint(ckp_path, run_variables=None, **kwargs):
    if not os.path.isfile(ckp_path):
        logger.error("=> No previous checkpoint found at '{}'".format(ckp_path))
        return

    logger.info("Found checkpoint at {}".format(ckp_path))
    checkpoint = torch.load(ckp_path, map_location="cpu", weights_only=False)

    for key, value in kwargs.items():
        if key in checkpoint and value is not None:
            try:
                msg = value.load_state_dict(checkpoint[key], strict=False)
                logger.info(
                    "=> loaded '{}' from checkpoint '{}' with msg {}".format(
                        key, ckp_path, msg
                    )
                )
            except TypeError:
                try:
                    msg = value.load_state_dict(checkpoint[key])
                    logger.info(
                        "=> loaded '{}' from checkpoint: '{}'".format(key, ckp_path)
                    )
                except ValueError:
                    logger.info(
                        "=> failed to load '{}' from checkpoint: '{}'".format(
                            key, ckp_path
                        )
                    )
        else:
            logger.info(
                "=> key '{}' not found in checkpoint: '{}'".format(key, ckp_path)
            )

    if run_variables is not None:
        for var_name in run_variables:
            if var_name in checkpoint:
                run_variables[var_name] = checkpoint[var_name]


def load_from_pretrained(model, ckpt_fname, ckpt_model_key="model", strict=True):
    if not os.path.isfile(ckpt_fname):
        logger.error("=> No pretrained model found at '{}'".format(ckpt_fname))
        return

    ckpt = torch.load(ckpt_fname, "cpu", weights_only=False)

    weights = {
        k.replace("module.", "").replace("_orig_mod.", ""): v
        for k, v in ckpt[ckpt_model_key].items()
    }
    # check if position embedding needs to be resized
    resize_positional_embed(model, weights)

    # load pretrained weights
    msg = model.load_state_dict(weights, strict=strict)
    logger.info(
        "=> Loaded pretrained model from {} (trained for {}) with msg:\n{}".format(
            ckpt_fname, get_training_duration(ckpt), msg
        )
    )


def resize_positional_embed(model, weights, num_nonpatch_tokens=1):
    """
    Resizes the positional embeddings in the checkpoint (weights) if needed.
    Model should contain as encoder an instance of DINOv2.
    num_nonpatch_tokens: number of non-patch tokens in the position embeddings of the model, i.e. [CLS]
    """
    old_num_patches = weights["encoder.pos_embed"].shape[1] - num_nonpatch_tokens
    new_num_patches = model.encoder.pos_embed.data.shape[1] - num_nonpatch_tokens

    if old_num_patches == new_num_patches:
        return

    logger.info(
        "=> Model has {} patches, while checkpoint has {}.".format(
            new_num_patches,
            old_num_patches,
        )
    )
    # image_size = model.encoder.patch_embed.img_size
    # patch_size = model.encoder.patch_size
    embed_dim = model.encoder.pos_embed.data.shape[-1]

    # unpack CLS and patch tokens in the old position embeddings
    # then resize to the new position embeddings size
    old_pos_embed = weights["encoder.pos_embed"]
    oldp_cls = old_pos_embed[:, :num_nonpatch_tokens]
    oldp_patches = old_pos_embed[:, num_nonpatch_tokens:]

    # assume square images
    old_num_patches_side = int(math.sqrt(old_num_patches))
    new_num_patches_side = int(math.sqrt(new_num_patches))

    # new position embeddings for patches
    newp_patches = (
        torch.nn.functional.interpolate(
            oldp_patches.reshape(
                1, old_num_patches_side, old_num_patches_side, embed_dim
            ).permute(0, 3, 1, 2),
            size=(new_num_patches_side, new_num_patches_side),
            mode="bicubic",
        )
        .permute(0, 2, 3, 1)
        .reshape(1, new_num_patches, embed_dim)
    )
    new_pos_embed = torch.cat((oldp_cls, newp_patches), dim=1)
    weights["encoder.pos_embed"] = new_pos_embed

    logger.info(
        "=> Positional embeddings in the checkpoint has been reshaped to {}".format(
            weights["encoder.pos_embed"].data.shape
        )
    )


def check_loss(loss):
    """Checks if loss is finite across all processes."""
    loss_is_finite = torch.tensor(
        [float(math.isfinite(loss.item()))], device=loss.device
    )
    dist.all_reduce(loss_is_finite, op=dist.ReduceOp.MIN)

    if loss_is_finite.item() == 1.0:
        # all good
        return

    if not math.isfinite(loss.item()):
        # This process has the NaN loss
        logger.info(f"Error: Loss is {loss.item()}, stopping training")
    else:
        # Other processes detect that at least one process has NaN loss
        logger.info(
            f"Process {get_global_rank()} detected NaN loss in another process. Exiting."
        )

    # Synchronize all processes before exiting
    dist.barrier()
    sys.exit(1)


def get_training_duration(ckpt: dict) -> str:
    if "iter" in ckpt:
        dur = "{} iterations".format(ckpt["iter"])
    elif "epoch" in ckpt:
        dur = "{} epochs".format(ckpt["epoch"])
    else:
        dur = "unknown duration"
    return dur


@functools.lru_cache()
def configure_logger(
    level: int = logging.INFO,
    name: Optional[str] = None,
    output: Optional[str] = None,
):
    """
    Configure a logger.

    Adapted from Detectron2.

    Args:
        name: The name of the logger to configure.
        level: The logging level to use.
        output: A file name or a directory to save log. If None, will not save log file.
            If ends with ".txt" or ".log", assumed to be a file name.
            Otherwise, logs will be saved to `output/log.txt`.

    Returns:
        The configured logger.
    """

    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.propagate = False

    # Loosely match Google glog format:
    #   [IWEF]yyyymmdd hh:mm:ss.uuuuuu threadid file:line] msg
    # but use a shorter timestamp and include the logger name:
    #   [IWEF]yyyymmdd hh:mm:ss logger threadid file:line] msg
    fmt_prefix = (
        "%(levelname).1s%(asctime)s %(process)s %(name)s %(filename)s:%(lineno)s] "
    )
    fmt_message = "%(message)s"
    fmt = fmt_prefix + fmt_message
    datefmt = "%Y-%m-%d %H:%M:%S"
    formatter = logging.Formatter(fmt=fmt, datefmt=datefmt)

    # stdout logging for main worker only
    if is_main_process():
        handler = logging.StreamHandler(stream=sys.stdout)
        handler.setLevel(logging.DEBUG)
        handler.setFormatter(formatter)
        logger.addHandler(handler)

    # file logging for all workers
    if output:
        if os.path.splitext(output)[-1] in (".txt", ".log"):
            filename = output
        else:
            filename = os.path.join(output, "logs", "log.txt")

        if not is_main_process():
            global_rank = get_global_rank()
            filename = filename + ".rank{}".format(global_rank)

        os.makedirs(os.path.dirname(filename), exist_ok=True)

        handler = logging.StreamHandler(open(filename, "a"))
        handler.setLevel(logging.DEBUG)
        handler.setFormatter(formatter)
        logger.addHandler(handler)

    return logger


class ExternalLogger(object):
    """
    Class to handle logging via external loggers such as Tensorboard.
    """

    def __init__(self, output_dir):
        self.output_dir = output_dir
        self.tb_enabled = False
        self.tb_writer = None
        self.tb_dir = None
        self._init_tb_logger()

    def _init_tb_logger(self):
        self.tb_enabled = is_main_process()
        if not self.tb_enabled:
            logger.info("Tensorboard is disabled")
            return

        self.tb_dir = os.path.join(self.output_dir, "tb")
        os.makedirs(self.tb_dir, exist_ok=True)
        logger.info("Tensorboard directory: {}".format(self.tb_dir))
        self.tb_writer = SummaryWriter(self.tb_dir, flush_secs=30)

    def log(
        self,
        stats: dict,
        step: int,
        prefix: str = "",
        save_path: str = "",
    ):
        if is_main_process() and save_path != "":
            with open(save_path, mode="a") as f:
                f.write(json.dumps(stats) + "\n")

        if prefix != "":
            stats = {prefix + k: v for k, v in stats.items()}

        if self.tb_enabled:
            for k, v in stats.items():
                self.tb_writer.add_scalar(k, v, step)


class MetricLogger(object):
    def __init__(self, delimiter="\t", output_file=None):
        self.delimiter = delimiter
        self.output_file = output_file
        self.meters = defaultdict(SmoothedValue)

    def update(self, **kwargs):
        for k, v in kwargs.items():
            if isinstance(v, torch.Tensor):
                v = v.item()
            assert isinstance(v, (float, int))
            self.meters[k].update(v)

    def __getattr__(self, attr):
        if attr in self.meters:
            return self.meters[attr]
        if attr in self.__dict__:
            return self.__dict__[attr]
        raise AttributeError(
            "'{}' object has no attribute '{}'".format(type(self).__name__, attr)
        )

    def __str__(self):
        loss_str = []
        for name, meter in self.meters.items():
            loss_str.append("{}: {}".format(name, str(meter)))
        return self.delimiter.join(loss_str)

    def synchronize_between_processes(self):
        for meter in self.meters.values():
            meter.synchronize_between_processes()

    def add_meter(self, name, meter):
        self.meters[name] = meter

    def reset(self):
        for meter in self.meters.values():
            meter.reset()

    def dump_in_output_file(self, iteration):
        if self.output_file is None or not is_main_process():
            return

        dict_to_dump = dict(
            iteration=iteration,
            time_iter=self.meters["time_iter"],
            time_data=self.meters["time_data"],
        )
        dict_to_dump.update({k: v.median for k, v in self.meters.items()})
        with open(self.output_file, "a") as f:
            f.write(json.dumps(dict_to_dump) + "\n")

    def log_every(
        self,
        iterable,
        print_freq,
        header=None,
        n_iterations=None,
        start_iteration=0,
        ext_logger: ExternalLogger = None,
        ext_logger_prefix="",
        reset_after_print=False,
    ):
        if not header:
            header = ""

        if n_iterations is None:
            n_iterations = len(iterable)

        if start_iteration >= n_iterations:
            logger.warning(
                "{} Nothing to do: start_iteration ({}) ≥ n_iterations ({})",
                header,
                start_iteration,
                n_iterations,
            )
            return

        space_fmt = ":" + str(len(str(n_iterations))) + "d"

        log_list = [
            header,
            "[{0" + space_fmt + "}/{1}]",
            "eta: {eta}",
            "{meters}",
            "max mem: {memory:.0f}",
        ]

        log_msg = self.delimiter.join(log_list)
        MB = 1024.0 * 1024.0
        it = start_iteration

        self.reset()
        start_time = time.time()
        end = time.time()

        for obj in iterable:
            self.meters["time_data"].update(time.time() - end)
            yield obj
            self.meters["time_iter"].update(time.time() - end)

            if it % print_freq == 0 or it == n_iterations - 1:
                self.synchronize_between_processes()

                self.dump_in_output_file(iteration=it)

                eta_seconds = self.meters["time_iter"].global_avg * (n_iterations - it)
                eta_string = str(datetime.timedelta(seconds=int(eta_seconds)))

                logger.info(
                    log_msg.format(
                        it,
                        n_iterations,
                        eta=eta_string,
                        meters=str(self),
                        memory=(
                            torch.cuda.max_memory_allocated() / MB
                            if torch.cuda.is_available()
                            else 0
                        ),
                    )
                )

                # Log via external logger
                if ext_logger is not None:
                    meters = {k: v.global_avg for k, v in self.meters.items()}
                    ext_logger.log(
                        stats=meters,
                        step=it,
                        prefix=ext_logger_prefix,
                    )

                if reset_after_print:
                    self.reset()

            it += 1
            end = time.time()
            if it >= n_iterations:
                break

        total_time = time.time() - start_time
        total_time_str = str(datetime.timedelta(seconds=int(total_time)))
        logger.info(
            "{} Total time: {} ({:.6f} s / it)".format(
                header, total_time_str, total_time / n_iterations
            )
        )
