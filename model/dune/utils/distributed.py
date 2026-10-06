import os
import sys

import torch
import torch.distributed as dist


def is_enabled() -> bool:
    return dist.is_available() and dist.is_initialized()


def get_global_size() -> int:
    return dist.get_world_size() if is_enabled() else 1


def get_global_rank() -> int:
    return dist.get_rank() if is_enabled() else 0


def is_main_process() -> bool:
    return get_global_rank() == 0


def init_distributed_mode(args):
    # launched with torchrun
    if "WORLD_SIZE" in os.environ:
        args.world_size = int(os.environ["WORLD_SIZE"])

        if "RANK" in os.environ:
            args.rank = int(os.environ["RANK"])
        elif "SLURM_PROCID" in os.environ:
            args.rank = int(os.environ["SLURM_PROCID"])
        else:
            print("Cannot find rank in environment variables")
            sys.exit(-1)

        n_gpus_per_node = torch.cuda.device_count()
        assert n_gpus_per_node > 0, "No GPU device detected"

        args.gpu = args.rank - n_gpus_per_node * (args.rank // n_gpus_per_node)

    # launched naively with python
    elif torch.cuda.is_available():
        print("==> Will run the code on one GPU.")
        args.rank, args.gpu, args.world_size = 0, 0, 1
        os.environ["MASTER_ADDR"] = "127.0.0.1"
        os.environ["MASTER_PORT"] = "12345"

    else:
        print("==> Does not support training without GPU.")
        sys.exit(1)

    print(
        "=> WORLD_SIZE={}, RANK={}, GPU={}, MASTER_ADDR={}, MASTER_PORT={}, INIT_METHOD={}".format(
            args.world_size,
            args.rank,
            args.gpu,
            os.environ["MASTER_ADDR"],
            os.environ["MASTER_PORT"],
            args.dist_url,
        ),
        flush=True,
    )

    dist.init_process_group(
        backend="nccl",
        init_method=args.dist_url,
        world_size=args.world_size,
        rank=args.rank,
    )
    dist.barrier()
    torch.cuda.set_device(args.gpu)
    setup_for_distributed(args.rank == 0)


def setup_for_distributed(is_master):
    """
    This function disables printing when not in master process
    """
    import builtins as __builtin__

    builtin_print = __builtin__.print

    def print(*args, **kwargs):
        force = kwargs.pop("force", False)
        if is_master or force:
            builtin_print(*args, **kwargs)

    __builtin__.print = print
