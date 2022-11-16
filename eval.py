import torch
from dataset_utils.configs import get_config
from evaluation.metrics import GT_ImageBasedRetrievalAccuracyCalculator
from models.model_factory import create_model
from dataset_utils.configs import get_config
import vsd_utils as utils
from dataset_utils.image_dataset import get_dataset_factory, create_model_transforms
import os
import fire
import pandas as pd
from dataset_utils.transforms import create_model_transforms

def eval(
        model_name: str,
        task='in_catalog_retrieval_zero_shot', 
        dataset='vsd-benchmark/vsd-fashion', 
        model_path=None, 
        img_size=224, 
        batch_size=840, 
        num_workers=8, 
        gpu_num=0, 
        seed=0
):
    utils.create_logger(os.path.abspath(__file__), dump=False)
    utils.set_initial_random_seed(seed)

    dataset_metadata = get_config(task, dataset)

    dataset_factory = get_dataset_factory(dataset_metadata=dataset_metadata,
                                        image_size=img_size,
                                        rnd_state=seed)

    transforms = create_model_transforms(model_name, img_size)

    query_loader, gallery_loader = dataset_factory.get_train_and_query_gallery_data_loaders(batch_size=batch_size,
                                                                                            num_workers=num_workers,
                                                                                            transforms=transforms)                                                                                                          
    
    with torch.inference_mode():
        device = torch.device(gpu_num if torch.cuda.is_available() else 'cpu')

        model = create_model(model_name, device, model_path)
        model = model.to(device)
        model.eval()                        

        gt_image_retrieval_acc = GT_ImageBasedRetrievalAccuracyCalculator(path_to_gt_json=dataset_metadata.annotations_file,
                                                                        ks_hr=[5, 9],
                                                                        ks_mrr=[5, 9],
                                                                        device=device,
                                                                        distance_func_name='cosine')


        mean_acc_at_k_gt, mrr_at_k, mrr, roc_auc, pr_auc = gt_image_retrieval_acc.calc(model=model,
                                                                    query_loader=query_loader,
                                                                    gallery_loader=gallery_loader,
                                                                    dataset_metadata=dataset_metadata,
                                                                    backbone_name=model_name,
                                                                    verbose=True)

    metrics = {}
    metrics['model'] = model_name

    for k, v in mean_acc_at_k_gt.items():
        metrics[f'HR@{k}'] = v['mean'] * 100
    for k, v in mrr_at_k.items():
        metrics[f'MRR@{k}'] = v['mean'] * 100

    metrics['ROC-AUC'] = roc_auc * 100
    metrics['PR-AUC'] = pr_auc * 100

    print(pd.DataFrame([metrics]))

if __name__ == '__main__':
    fire.Fire(eval)