import pytest
import torch
from muvi import get_free_gpu_idx


def test_early_stopping_callback():
    if torch.cuda.is_available():
        gpu_idx = get_free_gpu_idx()
        assert gpu_idx >= 0
    else:
        with pytest.raises(RuntimeError):
            get_free_gpu_idx()
