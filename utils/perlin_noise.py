'''
partially copied from https://gist.github.com/vadimkantorov/ac1b097753f217c5c11bc2ff396e0a57
'''
import copy

import torch
import math
import numpy as np

def rand_perlin_2d(shape, res, fade=lambda t: 6 * t ** 5 - 15 * t ** 4 + 10 * t ** 3):
    delta = (res[0] / shape[0], res[1] / shape[1])
    d = (shape[0] // res[0], shape[1] // res[1])

    grid = torch.stack(torch.meshgrid(torch.arange(0, res[0], delta[0]), torch.arange(0, res[1], delta[1]), indexing='ij'), dim=-1) % 1
    angles = 2 * math.pi * torch.rand(res[0] + 1, res[1] + 1)
    gradients = torch.stack((torch.cos(angles), torch.sin(angles)), dim=-1)

    tile_grads = lambda slice1, slice2: gradients[slice1[0]:slice1[1], slice2[0]:slice2[1]].repeat_interleave(d[0],
                                                                                                              0).repeat_interleave(
        d[1], 1)
    dot = lambda grad, shift: (
                torch.stack((grid[:shape[0], :shape[1], 0] + shift[0], grid[:shape[0], :shape[1], 1] + shift[1]),
                            dim=-1) * grad[:shape[0], :shape[1]]).sum(dim=-1)

    n00 = dot(tile_grads([0, -1], [0, -1]), [0, 0])
    n10 = dot(tile_grads([1, None], [0, -1]), [-1, 0])
    n01 = dot(tile_grads([0, -1], [1, None]), [0, -1])
    n11 = dot(tile_grads([1, None], [1, None]), [-1, -1])
    t = fade(grid[:shape[0], :shape[1]])
    return math.sqrt(2) * torch.lerp(torch.lerp(n00, n10, t[..., 0]), torch.lerp(n01, n11, t[..., 0]), t[..., 1])


def rand_perlin_2d_octaves(shape, res, octaves=1, persistence=0.5):
    h, w = shape
    h_ = h // 32
    if h%32 != 0:
        h_+=1
    h_ = h_*32
    w_ = w // 32
    if w % 32 != 0:
        w_ += 1
    w_ = w_ * 32
    noise = torch.zeros((h_, w_))
    frequency = 1
    amplitude = 1
    for _ in range(octaves):
        noise += amplitude * rand_perlin_2d((h_, w_), (frequency * res[0], frequency * res[1]))
        frequency *= 2
        amplitude *= persistence
    noise = noise[:h, :w]
    return noise


def binary(noise, t: float = 0.1):
    area = noise.shape[0] * noise.shape[1]
    m = torch.min(noise)
    if m < 0:
        noise += torch.abs(m)
    uni, counts = torch.unique(noise, return_counts=True)
    idx = torch.argmin(torch.abs((torch.cumsum(counts, dim=0) / area) - (1-t)))
    noise = noise > uni[idx]
    #noise[noise <= uni[idx]] = 0
    #noise[noise > uni[idx]] = 1
    #print(torch.sum(noise) / area)

    return noise

def random_binary_perlin_noise(shape=(256, 256), t=0.5, octs = [2, 4, 8, 16, 32]):
    oct1 = octs[int(np.random.randint(0, 4))]
    oct2 = octs[int(np.random.randint(0, 4))]
    persistence = int(np.random.randint(1, 10))
    if np.random.rand() < 0.5:
        noise = rand_perlin_2d_octaves(shape, (oct1, oct2), 1, persistence)
    else:
        noise = rand_perlin_2d(shape, (oct1, oct2))
    return binary(noise, t)


if __name__ == '__main__':
    import matplotlib.pyplot as plt
    x = 4
    ts = [0.9, 0.75, 0.5]
    y = len(ts) + 1

    for i in range(x):
        t = float(np.round(np.random.rand()*0.5 + 0.25, 4))
        shape = (256, 256)
        octs = [2, 4, 8, 16, 32]
        oct1 = octs[int(np.random.randint(0, 4))]
        oct2 = octs[int(np.random.randint(0, 4))]

        if np.random.rand() < 0.5:
            noise = rand_perlin_2d_octaves(shape, (oct1, oct2), 1, 100)
        else:
            noise = rand_perlin_2d(shape, (oct1, oct2))

        plt.subplot(x, y, i * y + 1)
        plt.imshow(noise, cmap='gray', interpolation='lanczos')
        plt.axis('off')
        if i == 0:
            plt.title('Perlin Noise'.format(t))

        for j, t in enumerate(ts):
            b_noise = binary(copy.deepcopy(noise), t)
            uni, counts = np.unique(b_noise == 0, return_counts=True)
            print(uni, [c/sum(counts) for c in counts])

            plt.subplot(x, y, i * y + 2 + j)
            plt.imshow(b_noise, cmap='gray', interpolation='lanczos')
            plt.axis('off')
            if i == 0:
                plt.title('Binary {}%'.format(int(np.round(t*100, 0))))
    plt.show()