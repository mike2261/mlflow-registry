"""Tiny device helper shared by the torch-based wrappers."""


def cuda_available() -> bool:
    try:
        import torch

        return torch.cuda.is_available()
    except ImportError:
        return False


def torch_device() -> str:
    return "cuda" if cuda_available() else "cpu"
