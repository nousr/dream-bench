import torch.nn as nn
from torch import no_grad


class NullSafety(nn.Module):
    def __init__(self):
        super().__init__()
        self.device = "cuda"

    @no_grad()
    def forward(self, clip_input, images):
        return images, [False for _ in range(len(images))]
