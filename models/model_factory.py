from .dino import Dino
from .argus import Argus
from .clip import CLIP
from .resnet import ResNet
from .beit import BEiT

paper_models = ['argus', 'dino', 'beit', 'clip']

def create_model(model_name, device, weights=None):
    if model_name == 'argus':
        return Argus()
    elif model_name == 'dino':
        return Dino()
    elif model_name == 'beit':
        return BEiT()
    elif model_name == 'resnet':
        return ResNet()
    elif model_name == 'clip':
        return CLIP(device)
    else:
        exit('bad model name')