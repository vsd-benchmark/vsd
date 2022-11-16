from PIL import Image, ImageFile
from functools import partial

import torch
import numpy as np
import os
import fsutil
from torch.utils.data import Dataset, DataLoader
import logging

from .transforms import dummy_transform, double_dummy_transform, create_model_transforms

ImageFile.LOAD_TRUNCATED_IMAGES = True

IMAGES_FIELD_NAME     = "images"
PATH_FIELD_NAME       = "path"
BBOX_FIELD_NAME       = "bbox"
ATTRIBUTES_FIELD_NAME = 'attributes'
ID_FIELD_NAME         = 'id'
PHASE_FIELD_NAME      = 'phase'
NUM_OF_ATTRIBUTES     = 1000

class ImageDataset(Dataset):
    """
    Image dataset that is based on a JSON metadata file. The metadata file is an array where each entry includes the details
    of an image in the dataset. The image details should include the relative path to the image from the dataset directory
    and any other wanted fields.
    """
    def __init__(self, dataset_metadata,
                 crop_bbox=False, im_size=224, with_metadata_transform=None, transform=None, metadata_transform=None):

        self.root_dir = dataset_metadata.images_folder
        self.metadata_path = dataset_metadata.metadata_file
        self.im_size = im_size
        self.crop_bbox = crop_bbox

        self.with_metadata_transform = with_metadata_transform if with_metadata_transform else double_dummy_transform
        self.transform = transform if transform else dummy_transform
        self.metadata_transform = metadata_transform if metadata_transform else dummy_transform

        self.seeds_file = dataset_metadata.seeds_file

        self.images_metadata = fsutil.read_file_json(self.metadata_path)[IMAGES_FIELD_NAME]
        self.test_files = [x['path'] for x in fsutil.read_file_json(dataset_metadata.manifest_file)[IMAGES_FIELD_NAME]]

        if self.seeds_file:
            self.seeds = fsutil.read_file_json(dataset_metadata.seeds_file)['seeds']
        else:
            annotations = fsutil.read_file_json(dataset_metadata.annotations_file)
            self.seeds = [item['key'][0] for item in annotations]
            
        for image_meta in self.images_metadata:
            image_meta['is_seed'] = image_meta['path'] in self.seeds
            image_meta['is_test'] = not self.test_files or image_meta['path'] in self.test_files

    def __preprocess_image(self, image, image_metadata):
        if len(image.mode) == 1:
            image = Image.fromarray(np.dstack((image, image, image)))

        if len(image.mode) > 3:
            image = Image.fromarray(np.array(image)[:, :, :3])

        if self.crop_bbox:
            bbox = image_metadata[BBOX_FIELD_NAME]
            x1 = int(bbox["x1"])
            y1 = int(bbox["y1"])
            x2 = int(bbox["x2"])
            y2 = int(bbox["y2"])
            return image.crop((x1, y1, x2, y2))
        else:
            return image

    def __preprocess_image_transform(self, image, metadata):
        image = self.with_metadata_transform(image, metadata)
        image = self.transform(image)
        return image

    def __getitem__(self, index):
        image_metadata = self.images_metadata[index]
        path = image_metadata[PATH_FIELD_NAME]

        image_metadata['is_seed'] = path in self.seeds
        image_metadata['is_test'] = not self.test_files or path in self.test_files

        image = Image.open(os.path.join(self.root_dir, path)).convert('RGB')
        image = self.__preprocess_image(image, image_metadata)

        return path, self.__preprocess_image_transform(image, image_metadata), self.metadata_transform(image_metadata)

    def __len__(self):
        return len(self.images_metadata)

class FieldOptionsMapper:
    """
    Maps a given index to the indices of all items with the same field value.
    """

    def __init__(self, metadatas, field_name, allow_missing_field=False):
        self.metadatas = metadatas
        self.field_name = field_name
        self.field_value_to_indices = {}
        self.allow_missing_field = allow_missing_field

        for index, metadata in enumerate(self.metadatas):
            field_value = self.__get_metadata_field_value(metadata)
            if field_value not in self.field_value_to_indices:
                self.field_value_to_indices[field_value] = []

            self.field_value_to_indices[field_value].append(index)

    def __get_metadata_field_value(self, metadata):
        if self.field_name in metadata:
            return metadata[self.field_name]
        elif self.allow_missing_field:
            return ""
        else:
            raise ValueError(f"Missing field value for field '{self.field_name}' in metadata: {json.dumps(metadata, indent=2)}")

    def __getitem__(self, index):
        field_value = self.get_field_value(index)
        return self.field_value_to_indices[field_value]

    def by_field_indices(self):
        return self.field_value_to_indices.values()

    def get_field_value(self, index):
        return self.__get_metadata_field_value(self.metadatas[index])

    def get_identifier(self, index):
        return self.get_field_value(index)

    def get_identifier_to_indices_dict(self):
        return self.field_value_to_indices
class SubsetImageDataset(torch.utils.data.Dataset):
    """
    Subset wrapper for ImageDataset. Uses only subset of the dataset with the given indices.
    """

    def __init__(self, image_dataset, indices, transform=None):
        self.image_dataset = image_dataset
        self.original_indices = indices
        self.transform = transform if transform else dummy_transform
        self.images_metadata = [self.image_dataset.images_metadata[i] for i in indices]
        self.labels = None

    def __getitem__(self, index):
        if self.labels is not None:
            out = self.transform(self.image_dataset[self.original_indices[index]])
            return out[0], out[1], self.labels[index]
        return self.transform(self.image_dataset[self.original_indices[index]])

    def __len__(self):
        return len(self.images_metadata)


class FilteredImageDataset(SubsetImageDataset):
    """
    Filter wrapper for ImageDataset. Filters such the dataset contains only matching images (match is done according to their metadata).
    """

    def __init__(self, image_dataset, filter, transform=None):
        self.image_dataset = image_dataset
        self.filter = filter
        relevant_indices = self.__create_relevant_images_indices()
        super().__init__(image_dataset, relevant_indices, transform if transform else dummy_transform)

    def __create_relevant_images_indices(self):
        relevant_images_indices = []
        for index, image_metadata in enumerate(self.image_dataset.images_metadata):
            if self.filter(image_metadata):
                relevant_images_indices.append(index)

        return relevant_images_indices
    

def create_frequent_values_filtered_image_dataset(image_dataset, field_name, freq_threshold=2):
    """
    Creates a FilteredImageDataset, filtering out images that have infrequent values for a certain field. The resulting dataset will contain only
    images that their value for the field has frequency that is greater than (or equals) to freq_threshold.
    :param image_dataset: image dataset.
    :param field_name: name of the field to filter according to its frequency.
    :param freq_threshold: threshold frequency to keep images with values that their frequency is above (or equal) to the threshold.
    :return: FilteredImageDataset with images with frequent values for the given field.
    """
    field_indices_mapper = FieldOptionsMapper(image_dataset.images_metadata, field_name)
    indices_to_remove = set()
    for field_value, indices in field_indices_mapper.field_value_to_indices.items():
        if len(indices) < freq_threshold:
            indices_to_remove.update(indices)

    indices_to_keep = [i for i in range(len(image_dataset)) if i not in indices_to_remove]
    return SubsetImageDataset(image_dataset, indices_to_keep)

def filter_func(field_name, field_values, image_metadata):
    return image_metadata[field_name] in field_values


def create_field_filtered_image_dataset(image_dataset, field_name, field_values):
    """
    Creates an image dataset that is a subset of the given dataset with only the images that have a matching value for the given field name and
    values.
    :param image_dataset: image dataset.
    :param field_name: name of the field to filter by.
    :param field_values: sequence of matching field values. An item will be in the filtered dataset if its field value is in this sequence.
    :return: FilteredImageDataset with only the images with a field value that is in the given field values for the field name.
    """

    return FilteredImageDataset(image_dataset, partial(filter_func, field_name, field_values))


class MultiTaskDatasetFactory():
    def __init__(self, 
                 dataset_metadata,
                 crop_bbox=False,
                 image_size=224,
                 in_memory=False,
                 id_freq_threshold=1,
                 rnd_state=None):

        self.dataset_metadata = dataset_metadata
        self.crop_bbox = crop_bbox
        self.image_size = image_size
        self.in_memory = in_memory
        self.id_freq_threshold = id_freq_threshold
        
        if rnd_state != None:
            self.rnd_state = np.random.RandomState(rnd_state)
        else:
            self.rnd_state = rnd_state


    def get_train_and_query_gallery_datasets(self, transform):

        image_dataset = ImageDataset(dataset_metadata=self.dataset_metadata,
                                     im_size=self.image_size,
                                     transform=transform)
        
        dataset = create_field_filtered_image_dataset(image_dataset, "phase", ["train", "test", "query", "gallery", "search"])

        dataset = create_frequent_values_filtered_image_dataset(dataset, "id", self.id_freq_threshold)

        test_dataset = create_field_filtered_image_dataset(dataset, 'is_test', [True])

        query_dataset = create_field_filtered_image_dataset(test_dataset, 'is_seed', [True])

        # If the task metadata specifics that the gallery should be used from specific phases in the dataet, filter the phases.
        # Relevant mainly for the wild zero shot task.
        if hasattr(self.dataset_metadata, 'gallery_phases'):
            gallery_dataset = create_field_filtered_image_dataset(test_dataset, 'phase', self.dataset_metadata.gallery_phases)
        else:
            # If not, use the entire dataset.
            gallery_dataset = test_dataset
        
        return query_dataset, gallery_dataset
    
    def get_train_and_query_gallery_data_loaders(self, batch_size=128, num_workers=0, transforms=create_model_transforms(None), shuffle_test=False):
        query_dataset, gallery_dataset = self.get_train_and_query_gallery_datasets(transforms)
        logging.info('Created dataset, query has {} images, gallery has {} images'.format(
            len(query_dataset), len(gallery_dataset)))

        query_loader = DataLoader(query_dataset, batch_size=batch_size, num_workers=num_workers, shuffle=shuffle_test)
        gallery_loader = DataLoader(gallery_dataset, batch_size=batch_size, num_workers=num_workers, shuffle=shuffle_test)
        return query_loader, gallery_loader


def get_dataset_factory(dataset_metadata, crop_bbox=False, image_size=224, in_memory=False, id_freq_threshold=1, rnd_state=None):
        return MultiTaskDatasetFactory(dataset_metadata, crop_bbox, image_size, in_memory, id_freq_threshold, rnd_state)