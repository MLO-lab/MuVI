import torch
from muvi import get_free_gpu_idx


def test_get_free_gpu_idx():
    if torch.cuda.is_available():
        gpu_idx = get_free_gpu_idx()
        assert gpu_idx >= 0
