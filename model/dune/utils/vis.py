import os

import numpy as np
import matplotlib.pyplot as plt
from torchvision.utils import save_image


def save_batched_image(image, save_path, nrow=8, padding=3, normalize=True):
    """
    Save a batch of images to a grid.
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    kwargs = {
        "nrow": nrow,
        "padding": padding,
        "normalize": normalize,
        "scale_each": normalize,
        "pad_value": 255,
    }

    save_image(image, save_path, **kwargs)


def plot_arr(arr: np.ndarray, save_path: str, dpi: int = 300):
    plt.close()
    plt.figure()
    plt.plot(arr)
    plt.grid()
    plt.savefig(save_path, dpi=dpi, bbox_inches="tight")
    np.save(save_path.replace(".png", ".npy"), arr)
