import torch


def get_params_groups(model, save_file_path=None):
    """
    Returns two parameters group, one for regularized parameters with weight decay,
    and another for unregularized parameters.
    """
    regularized = []
    not_regularized = []

    fp = None
    if save_file_path is not None:
        fp = open(save_file_path, "w")

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue

        if name.endswith(".bias") or len(param.shape) == 1:
            regstat = "Not Regularized"
            not_regularized.append(param)
        else:
            regstat = "Regularized"
            regularized.append(param)

        if fp is not None:
            fp.write("{} - {} - {}\n".format(name, list(param.shape), regstat))

    if fp is not None:
        fp.flush()
        fp.close()

    return [{"params": regularized}, {"params": not_regularized, "weight_decay": 0.0}]


def clip_gradients(model, clip):
    norms = []
    for n, p in model.named_parameters():
        if p.grad is None:
            continue

        param_norm = p.grad.data.norm(p=2)
        norms.append(param_norm)
        clip_coef = clip / (param_norm + 1e-6)
        if clip_coef < 1:
            p.grad.data.mul_(clip_coef)

    return torch.stack(norms)
