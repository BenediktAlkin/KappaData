import torch
from pathlib import Path
from kappadata.utils.logging import get_log_or_pass_function

def setup(log_fn, dst, seed):
    log = get_log_or_pass_function(log_fn)
    dst = Path(dst).expanduser()
    assert not dst.exists(), f"{dst.as_posix()} already exists"
    dst.mkdir(parents=True)
    generator = torch.Generator().manual_seed(seed)
    return log, dst, generator