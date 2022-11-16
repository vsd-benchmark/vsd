from torchvision import transforms
from PIL import Image, ImageFile

try:
    from torchvision.transforms import InterpolationMode
    BICUBIC = InterpolationMode.BICUBIC
except ImportError:
    BICUBIC = Image.BICUBIC

def _convert_image_to_rgb(image):
    return image.convert("RGB")

def create_model_transforms(model_name, im_size=224):
    if model_name == 'beit':
        return transforms.Compose([ transforms.Resize((im_size, im_size)),
                                    transforms.ToTensor(),
                                    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])])
    elif model_name == 'clip':
        return transforms.Compose([
            transforms.Resize(im_size, interpolation=BICUBIC),
            transforms.CenterCrop(im_size),
            _convert_image_to_rgb,
            transforms.ToTensor(),
            transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711))
        ])
    else:
        return transforms.Compose([transforms.Resize((im_size, im_size)),
                            transforms.ToTensor(),
                            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
    
def dummy_transform(x):
        return x

def double_dummy_transform(x, y):
    return x