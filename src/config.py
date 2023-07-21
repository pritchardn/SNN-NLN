import torch

WANDB_ACTIVE = False
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
