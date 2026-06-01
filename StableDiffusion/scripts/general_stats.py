import torch

def get_memory_used_torch():
     """Returns memory in MegaBytes."""
     return torch.cuda.memory_allocated() / (1024 ** 2)