import numpy as np
import torch
from skimage.measure import marching_cubes as _mc


def marching_cubes(level, threshold):
    verts, faces, _, _ = _mc(
        level.cpu().numpy(), level=threshold, gradient_direction="descent"
    )
    return torch.from_numpy(verts[:, ::-1].copy()), torch.from_numpy(
        faces.astype(np.int64).copy()
    )
