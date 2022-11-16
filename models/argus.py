
from torch import nn
import torch
import torch.nn.functional as F
from torchvision import models
import gdown
from pathlib import Path

ARGUS_CACHE_PATH = Path('./.cache/argus_densenet.pth')

class Argus(nn.Module):
    def __init__(self, backbone_name="argus"):
        super().__init__()

        if not ARGUS_CACHE_PATH.exists():
            ARGUS_CACHE_PATH.parent.mkdir(exist_ok=True, parents=True)
            gdown.download(id='1CeQnFHHEt9jFfhB8lONMNIywsDkOOiQi', output=str(ARGUS_CACHE_PATH))

        self.base_model = create_pretrained_cnn_backbone(backbone_name=backbone_name, requires_grad=True)
        self.g = nn.Sequential(nn.Linear(2208, 512, bias=True), nn.BatchNorm1d(512),
                               nn.ReLU(inplace=True), nn.Linear(512, 128, bias=True))

    def forward(self, x):
        y = self.base_model(x)
        return F.normalize(y, dim=-1)


    def unfreeze(self):
        for name, param in self.base_model.named_parameters():
            param.requires_grad = True


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
            
def create_pretrained_cnn_backbone(backbone_name: str, requires_grad: bool = True):
    if backbone_name == "argus":
        return create_argus_densenet161_module(ARGUS_CACHE_PATH, requires_grad=requires_grad)
    else:
        raise ValueError(f"Unsupported backbone name: {backbone_name}")

def create_argus_densenet161_module(model_path, requires_grad=True):
    model = models.densenet161()
    pretrained_weight = torch.load(model_path)['state_dict']
    load_argus_state_dict(model, pretrained_weight)
    model.classifier = nn.Sequential()

    for param in model.parameters():
        param.requires_grad = requires_grad

    return model


def load_argus_state_dict(model, pretrained_weights):
    weights = model.state_dict()

    # Remove non-exist keys
    for key in pretrained_weights.keys() - weights.keys():
        print("Delete unused model state key: %s" % key)
        del pretrained_weights[key]

    # Remove keys that size does not match
    for key, pretrained_weight in list(pretrained_weights.items()):
        weight = weights[key]
        if pretrained_weight.shape != weight.shape:
            print("Delete model state key with unmatched shape: %s" % key)
            del pretrained_weights[key]

    # Copy everything that pretrained_weights miss
    for key in weights.keys() - pretrained_weights.keys():
        print("Missing model state key: %s" % key)
        pretrained_weights[key] = weights[key]

    # Load the weights to model
    model.load_state_dict(pretrained_weights)