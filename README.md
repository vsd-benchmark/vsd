# Efficient Discovery and Effective Evaluation of Visual Similarities: A Benchmark and Beyond

[![arXiv](https://img.shields.io/badge/arXiv-2308.14753-b31b1b.svg)](https://arxiv.org/abs/2308.14753)

[![VSD Leaderboard](https://img.shields.io/badge/VSD-Leaderboard-blue)](https://vsd-benchmark.github.io/vsd/)

This repository contains the relevant data structures and code for evaluating models on the VSD task (including the models used in the paper).

---

How to run the benchmark
------------------------

1. Install python requirements:

```sh
pip install -r requirements.txt
```

2. Download dataset images explained in "Data Preparation"
3. Choose a task and a model to run, and invoke the evaluation script `eval.py` for example:

To evaluate the BeiT model in the 'in_catalog_retrieval_zero_shot' task run:

```sh
python eval.py --task in_catalog_retrieval_zero_shot --model_name beit
```

Supported tasks are:

1. in_catalog_retrieval_zero_shot
2. in_catalog_open_catalog
3. in_catalog_closed_catalog
4. consumer-catalog_wild_zero_shot

More information about each task can be found in the leaderboard site, and HuggingFace dataset [vsd-benchmark/vsd-fashion](https://huggingface.co/datasets/vsd-benchmark/vsd-fashion).

for more help run:

```sh
python eval.py --help
```

---

Supported models
----------------

This repository supports the models evaluated in the VSD paper:

1. BeiT
2. CLIP
3. ResNet
4. Dino
5. Argus

In order to add your own model, adjust `models/model_factory.py` to include your model and create it by your chosen name.
Then you can invoke the benchmark using:

```sh
python eval.py --task choosen task --model_name your_model_name
```

---

Data Preparation
----------------

The DeepFashion dataset can be downloaded from [here](https://mmlab.ie.cuhk.edu.hk/projects/DeepFashion.html).

The In-shop Clothes Retrieval Benchmark and Consumer-to-shop Clothes Retrieval Benchmark should be downloaded and extracted to ```datasets/img```. There should be six folders in ```datasets/img``` after extraction:

```
datasets/img/CLOTHING - Consumer-to-shop Clothes Retrieval Benchmark
datasets/img/DRESSES - Consumer-to-shop Clothes Retrieval Benchmark
datasets/img/TOPS - Consumer-to-shop Clothes Retrieval Benchmark
datasets/img/TROUSERS - Consumer-to-shop Clothes Retrieval Benchmark
datasets/img/MEN - In-shop Clothes Retrieval Benchmark
datasets/img/WOMEN - In-shop Clothes Retrieval Benchmark
```

---

Datasets
-----------

The VSD datasets annotations can be found and downloaded from our HuggingFace dataset [vsd-benchmark/vsd-fashion](https://huggingface.co/datasets/vsd-benchmark/vsd-fashion).
The evaluation script will download and use the relevant annotation files automatically, there is no need to download them yourself.

The VSD-Fashion dataset contains multiple tasks, each can be evaluated using this repository.

All the tasks share the following files:

1. Metadata file - Information about each image
2. Seeds file - The queries used in this task
3. Tags file - The similarity annotations of this task
4. Annotation images - Image files used in the task (exist only for tasks that have train/test split)

Tasks with train/test split will contain a set of files for each set.

The metadata json format is:

```
{
    'images': [  
                    {
                        'id': <image_id> (str) (mandatory),
                        'path': <image_relative_path> (str) (mandatory),
                        'phase': <image_train_test_fold_type> (str) (mandatory),
    
                        'bbox': <[x, y, h, w]> (list of ints) (optional),
                        'category': <object_cvategory> (str) (optional),
                        'color': <object_color> (str) (optional),
                    },
                ...
  
                ]
}
```

The metadata json contains a dict with a single mandatory key ```images```, its value is a list of dicts.
Each dict is an item - an image with different attributes.


```./datasets_utils/image_dataset``` contains the relevant classes and objects for initializing datasets and dataloaders
following the described data format. The main object is ```MultiTaskDatasetFactory```.

---

Evaluation and metrics
----------------------

Relevant metrics to monitor and validate:

- *Image based ground-truth metrics* : we used human annotators for manually tag ground truth labels for selected
  (seed, candidate) pairs. We produce (seed, candidate) from the topK predictions of a model. The pairs are given to human annotators for manual scoring of 0/1. The tagged pairs are presented in the attached zip and should be extracted into:
  ```datasets/gt_tagging/```

  The ground truth label jsons have the following format:

  ```
  [
      {"key": [<seed_image_name>, <candidate_image_name>], "value": <score>},
      {"key": [<seed_image_name>, <candidate_image_name>], "value": <score>},
      {"key": [<seed_image_name>, <candidate_image_name>], "value": <score>},
      ....

  ]
  ```

## Citation
If you find this useful, please cite our paper:
```
@inproceedings{barkan2023efficient,
  title={Efficient Discovery and Effective Evaluation of Visual Perceptual Similarity: A Benchmark and Beyond},
  author={Barkan, Oren and Reiss, Tal and Weill, Jonathan and Katz, Ori and Hirsch, Roy and Malkiel, Itzik and Koenigstein, Noam},
  booktitle={Proceedings of the IEEE/CVF International Conference on Computer Vision},
  pages={20007--20018},
  year={2023}
}
```

