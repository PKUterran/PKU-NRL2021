import numpy as np
import torch
from typing import Tuple


def set_seed(seed: int, use_cuda=False):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if use_cuda:
        torch.cuda.manual_seed(seed)


def get_mean_std(array: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    mean = torch.mean(array, dim=0, keepdim=True)
    std = torch.std(array - mean, dim=0, keepdim=True)
    return mean, std
