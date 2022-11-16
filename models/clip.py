
import numpy as np
import torch.nn.functional as F
import torch.nn as nn
import clip

class CLIP(nn.Module):
    def __init__(self, device):
        super().__init__()
        self.model, _ = clip.load("ViT-B/32", device=device)
        print("Model parameters:", f"{np.sum([int(np.prod(p.shape)) for p in self.model.parameters()]):,}")
        print("Context length:", self.model.context_length)
        print("Vocab size:", self.model.vocab_size)

    def forward(self, x):
        return F.normalize(self.model.encode_image(x), dim=-1)