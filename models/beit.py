
from torch import nn
import torch
import torch.nn.functional as F
from transformers import BeitModel

class BEiT(nn.Module):
    def __init__(self):
        super().__init__()
        self.base_model = BeitModel.from_pretrained('microsoft/beit-base-patch16-224-pt22k-ft22k')

    def forward(self, x):
        return self.base_model(x).pooler_output

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