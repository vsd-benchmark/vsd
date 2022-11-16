
import torch
import torch.nn.functional as F
import torch.nn as nn
from torchvision import models

class ResNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.base_model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)

    def forward(self, x):
        return F.normalize(self.base_model(x), dim=-1)

    def load_checkpoint(self, params):
        if params.checkpoint != None:
            trainer_state_dict = torch.load(params.checkpoint, map_location=params.device)
            if 'state_dict' in trainer_state_dict:
                print('Loaded checkpint from: {}'.format(params.checkpoint))
                self.load_state_dict(trainer_state_dict["state_dict"])
            elif 'model' in trainer_state_dict:
                print('Loaded checkpint from: {}'.format(params.checkpoint))
                self.load_state_dict(trainer_state_dict["model"])
            else:
                raise ValueError