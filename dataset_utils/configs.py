from datasets import load_dataset_builder
from types import SimpleNamespace

def get_config(task, dataset_name='vsd-benchmark/vsd-fashion', image_folder='./datasets'):
    dataset_builder = load_dataset_builder(dataset_name, task, image_folder=image_folder)
    splits = dataset_builder._split_generators(None)

    # Get HuggingFace Dataset split metadata
    test_set = next(s for s in splits if s.name == 'test')
    
    return SimpleNamespace(**test_set.gen_kwargs)

