import json
import torch
from tqdm import tqdm
import numpy as np
from abc import ABC
import json
import torch
from tqdm import tqdm
import numpy as np
import sklearn.metrics.pairwise
from sklearn.metrics import roc_auc_score, average_precision_score
import pandas as pd
from fsutil import read_file_json

def get_representations_from_model(model, data_loader, device, use_tqdm=True, model_name="", **kwargs):
    """
    Extract the representation from a model
    :param model: (torch.nn.Module)
    :param data_loader: (torch.data.Dataloader)
    :param device: (torch.device)
    :param use_tqdm: (bool), if set, presents a visualization of the progress of the FF process
    :return: name2embed (dict), map from the image name to the image's embedding
    """

    if model.training:
        model.eval()
        is_training = True
    else:
        is_training = False

    name2embed = {}
    enum = tqdm(data_loader) if use_tqdm else data_loader

    with torch.no_grad():
        for batch in enum:
            image_paths, images, metadata = batch

            images = images.to(device)
            embeds = model(images)
            embeds = embeds.detach().cpu()
            for image_path, e in zip(image_paths, embeds):
                name2embed[image_path] = e
    if is_training: model.train()
    return name2embed


def get_mapping_from_dataset_metadata(dataset_metadata, source_field, dest_field):
    """
    Extract a mapping from the fields in dataset_metadata
    """

    with open(dataset_metadata.metadata_file, 'r') as f:
        metadata = json.load(f)['images']

    mapping = {}
    for i, item in enumerate(metadata):
        source_key = i if source_field == 'ind' else item[source_field]
        mapping[source_key] = item[dest_field]
    return mapping

def get_mapping_from_dataset_metadata_id_to_paths(dataset_metadata):
    """
    Extract a mapping from the fields in dataset_metadata
    """

    metadata = read_file_json(dataset_metadata.metadata_file)['images']

    mapping = {}
    source_key = 'id'
    dest_key = 'path'
    for i, item in enumerate(metadata):
        if item[source_key] not in mapping.keys():
            mapping[item[source_key]] = []
        else:
            mapping[item[source_key]].append(item[dest_key])
    return mapping


def get_image_name_to_id_mapping(name2embed, dataset_metadata):
    path2id = get_mapping_from_dataset_metadata(dataset_metadata, 'path', 'id')
    image_name2_indx = {}
    id2_indxs = {}
    for sn, image_name in enumerate(name2embed.keys()):
        id = path2id[image_name]
        if id not in id2_indxs:
            id2_indxs[id] = []
        id2_indxs[id].append(sn)
        image_name2_indx[image_name] = sn

    indx2image_name = {v: k for k, v in image_name2_indx.items()}

    ind2id = {}
    for id, indxs in id2_indxs.items():
        for indx in indxs:
            ind2id[indx] = id

    return ind2id, indx2image_name


def cosine_similarity(tensors1, tensors2):
    """
    Given tensors1 of size (N, d) and tensors2 of shape (M, d), calculates and returns all pairwise cosine similarities in
    a tensor of size (N, M).
    """
    norms1 = tensors1.norm(p=2, dim=1, keepdim=True)
    norms2 = tensors2.norm(p=2, dim=1, keepdim=True)
    return torch.mm(tensors1 / norms1, (tensors2 / norms2).t())


def euclidean_distance(tensors1, tensors2):
    """
    Given tensors1 of size (N, d) and tensors2 of shape (M, d), calculates and returns all pairwise euclidean distances in
    a tensor of size (N, M).
    """
    orig_device = tensors1.device

    np_tensors1 = tensors1.detach().cpu().numpy()
    np_tensors2 = tensors2.detach().cpu().numpy()
    dist_mat = sklearn.metrics.pairwise.euclidean_distances(np_tensors1, np_tensors2)
    return torch.from_numpy(dist_mat).to(orig_device)


class BaseRetrievalAccuracyCalculator(ABC):
    """
    Abstract class that implements functions to be used by the different retrieval evaluators.
    """

    def __init__(self,  device, distance_func_name='cosine'):
        """
        :param device: (torch.device)
        :param distance_func_name: (str) the distance function name ['cosine', 'euclidean']
        """

        self.distance_func_name = distance_func_name
        self.distance_func = cosine_similarity if distance_func_name == 'cosine' else euclidean_distance
        self.device = device

    def get_topk_results(self, model, dataset_metadata, query_loader, gallery_loader, verbose=True, backbone_name="", largest=True):
        """
        Gets a model and two dataloaders and calculate the scores between every query items and the gallery items,
        results scores matrix (num_query X num_gallery). Then the function sorts each row (descending order)
        and reports the sorted scores matrix and the indices.
        """

        query_name2embed = get_representations_from_model(model, query_loader, self.device, verbose, model_name=backbone_name, set="query")

        self.query_ind2id, self.query_ind2image_path = get_image_name_to_id_mapping(query_name2embed, dataset_metadata)

        if gallery_loader is None:
            gallery_name2embed = query_name2embed
            self.gallery_ind2id = self.query_ind2id
            self.gallery_ind2image_path = self.query_ind2image_path
            filter_out_diag = True
        else:
            gallery_name2embed = get_representations_from_model(model, gallery_loader, self.device, verbose, model_name=backbone_name, set="gallery")
            self.gallery_ind2id, self.gallery_ind2image_path = get_image_name_to_id_mapping(gallery_name2embed, dataset_metadata)
            filter_out_diag = False


        self.id2query_ind = {v: k for k, v in self.query_ind2id.items()}
        self.id2gallery_ind = {v: k for k, v in self.gallery_ind2id.items()}
        self.image_path2query_ind = {v: k for k, v in self.query_ind2image_path.items()}
        self.image_path2gallery_ind = {v: k for k, v in self.gallery_ind2image_path.items()}

        # calc the topk similar items
        query_items = torch.stack(list(query_name2embed.values())).to(self.device)
        gallery_items = torch.stack(list(gallery_name2embed.values())).to(self.device)

        top_k_scores, top_k_inds = self._get_top_results(query_items, gallery_items,
                                                         num_results=len(gallery_items),
                                                         filter_out_diag=filter_out_diag,
                                                         largest=largest)
        top_k_inds = top_k_inds.detach().cpu().tolist()
        top_k_scores = top_k_scores.detach().cpu().tolist()
        return top_k_scores, top_k_inds

    def _get_top_results(self, query_tensors, tensors, num_results=1, largest=True, filter_out_diag=True):
        """
        Calculate the similarity between evert query tensor and the other tensors, report the topK by row.
        """
        metric_matrix = self.distance_func(query_tensors, tensors)
        if filter_out_diag:
            min_val = -1 if self.distance_func_name == 'cosine' else 0
            metric_matrix = metric_matrix.fill_diagonal_(min_val)
        if metric_matrix.shape[0] > 15000:
            metric_matrix = metric_matrix.cpu()
        return torch.topk(metric_matrix.to(dtype=torch.float32), min(num_results, metric_matrix.size(1)), dim=1, largest=largest)

    def filter_results_metadata_field(self, top_k_scores, top_k_inds, dataset_metadata, metadata_field):
        """
        Given topK scores and indices, filter them by a metadata_field. Meaning given a query item - get its
        metadata_field from the metadata, filter and keep gallery items with the same metadata_field.
        """

        path2metadata_field = get_mapping_from_dataset_metadata(dataset_metadata, 'path', metadata_field)
        query_ind2field = {ind: path2metadata_field[path] for ind, path in self.query_ind2image_path.items()}
        gallery_ind2field = {ind: path2metadata_field[path] for ind, path in self.gallery_ind2image_path.items()}

        filtered_top_k_inds = []
        filtered_top_k_scores = []

        for q_ind, (g_top_inds, g_top_scores) in enumerate(zip(top_k_inds, top_k_scores)):
            q_field = query_ind2field[q_ind]

            valid_g_top_inds = []
            valid_g_top_scores = []
            for g_ind, g_score in zip(g_top_inds, g_top_scores):
                if gallery_ind2field[g_ind] == q_field:
                    valid_g_top_inds.append(g_ind)
                    valid_g_top_scores.append(g_score)

            filtered_top_k_inds.append(valid_g_top_inds)
            filtered_top_k_scores.append(valid_g_top_scores)

        return filtered_top_k_scores, filtered_top_k_inds

    def filter_results_unique_ids(self, top_k_scores, top_k_inds, maxk, include_query_id):
        """
        The scores are being measured between images, each image is associated with a product id. A product id my have
        multiple images. This function filters the retrieved topk gallery items to contain only items with unique
        product ids. Meaning, if two images of the same product id were rated at location 1 and 2, the mechanism
        will drop the second item and will only keep the item that appears in location 1.
        """

        filtered_top_k_inds = []
        filtered_top_k_scores = []

        for q_ind, (g_top_inds, g_top_scores) in enumerate(zip(top_k_inds, top_k_scores)):
            valid_g_ids = [self.query_ind2id[q_ind]] if include_query_id else []

            valid_g_top_inds = []
            valid_g_top_scores = []

            for g_ind, g_score in zip(g_top_inds, g_top_scores):
                if self.gallery_ind2id[g_ind] not in valid_g_ids:
                    valid_g_top_inds.append(g_ind)
                    valid_g_top_scores.append(g_score)
                    valid_g_ids.append(self.gallery_ind2id[g_ind])
                    if len(valid_g_ids) > maxk:
                        break

            filtered_top_k_inds.append(valid_g_top_inds)
            filtered_top_k_scores.append(valid_g_top_scores)

        return filtered_top_k_scores, filtered_top_k_inds

    def get_maxk_results(self, top_k_scores, top_k_inds, maxk):
        filtered_top_k_inds = []
        filtered_top_k_scores = []
        for g_top_inds, g_top_scores in zip(top_k_inds, top_k_scores):
            filtered_top_k_inds.append(g_top_inds[:maxk])
            filtered_top_k_scores.append(top_k_scores[:maxk])

        return filtered_top_k_scores, filtered_top_k_inds


class GT_ImageBasedRetrievalAccuracyCalculator(BaseRetrievalAccuracyCalculator):
    """
    The ground truth tagging json format:
    [
        {"key": [<seed_image_name>, <candidate_image_name>], "value": <score>},
        {"key": [<seed_image_name>, <candidate_image_name>], "value": <score>},
        {"key": [<seed_image_name>, <candidate_image_name>], "value": <score>},
        ....
    ]
    """

    def __init__(self, path_to_gt_json, ks_hr, ks_mrr, device, distance_func_name='cosine'):
        super().__init__(device, distance_func_name)

        # self.protocol = path_to_gt_json.split('/')[-1].split('_')[0]

        self.ks_hr = ks_hr
        self.ks_mrr = ks_mrr

        with open(path_to_gt_json, 'r') as f:
            gt_tagging = json.load(f)
        
        self.gt_tagging = set([(tuple(item['key']) + (item['value'], )) for item in gt_tagging])

        self.labels = {frozenset((item[0], item[1])): item[2] for item in self.gt_tagging}

        self.num_tags = float(len(self.gt_tagging))
        print('Loaded {} positive tagging'.format(self.num_tags))

    def get_label(self, query, gal_id, labels=None):
        gallery = self.gallery_ind2image_path[gal_id]
        return self.labels.get(frozenset((query, gallery)), -1)

    def calc(self, model, dataset_metadata, query_loader, gallery_loader=None, include_query_id=True,
             backbone_name="", verbose=True, largest=True):

        """
        Calculate the retrieval metrics

        :param model: (torch.nn.Module)
        :param dataset_metadata: (Config)
        :param query_loader: (torch.data.Dataloader)
        :param gallery_loader: (torch.data.Dataloader)
        :param filter_by_metadata_field (bool) : if TRUE filter the retrieved recommendations list by a metadata field
        :param metadata_field (string) : the name of the metadata field to filter results
        :param verbose: (bool)
        :return: mean_hits_at_k (dict), mrr (dict), mrr_at_k (dict)
        """

        top_k_scores, top_k_inds = self.get_topk_results(model, dataset_metadata, query_loader, gallery_loader, verbose, backbone_name, largest=largest)
        self.metadata = dataset_metadata
        self.q_l = query_loader

        self.top_k_scores, self.top_k_inds = self.filter_results_unique_ids(top_k_scores, top_k_inds, max(self.ks_hr + self.ks_mrr), include_query_id=include_query_id)
            
        mean_hits_at_k = self._calc_mean_ht_at_k(self.top_k_inds)
        mrr = self._calc_mrr(self.top_k_inds)
        mrr_at_k = self._calc_mrr_at_k(self.top_k_inds)
        self.top_k_scores, self.top_k_inds = top_k_scores, top_k_inds
        roc_auc, pr_auc = self._calc_roc_pr_auc(top_k_scores ,top_k_inds)
        
        return mean_hits_at_k, mrr_at_k, mrr, roc_auc, pr_auc

    def _calc_mean_ht_at_k(self, top_k_inds):
        hits_at_k = {k: 0.0 for k in self.ks_hr}
        num_seeds = set()
        for (seed_fname, cand_fname, value) in self.gt_tagging:
            if value == 0:
                continue
            num_seeds.add(seed_fname)
            cand_ind = self.image_path2gallery_ind[cand_fname]
            seed_ind = self.image_path2query_ind[seed_fname]
            topk_preds = top_k_inds[seed_ind]
            topk_preds = [i for i in topk_preds if self.gallery_ind2id[i] != self.query_ind2id[seed_ind]]

            for k in self.ks_hr:
                if cand_ind in topk_preds[:k]:
                    hits_at_k[k] += 1.0

        hits_at_k_met = {k: {'mean': v / (k * len(num_seeds))} for k, v in hits_at_k.items()}
        return hits_at_k_met

    def _calc_mrr(self, top_k_inds):
        mrr = []
        num_seeds = set()
        for (seed_fname, cand_fname, value) in self.gt_tagging:
            if value == 0:
                continue
            num_seeds.add(seed_fname)
            seed_ind = self.image_path2query_ind[seed_fname]
            cand_ind = self.image_path2gallery_ind[cand_fname]
            topk_preds = top_k_inds[seed_ind]
            topk_preds = [i for i in topk_preds if self.gallery_ind2id[i] != self.query_ind2id[seed_ind]]
            try:
                rank = topk_preds.index(cand_ind) + 1.0
                mrr.append(1.0 / rank)
            except:
                mrr.append(0.0)

        mrr = np.array(mrr)
        mrr_met = {'mean': mrr.mean(), 'std': mrr.std()}
        return mrr_met

    def _calc_mrr_at_k(self, top_k_inds):
        mrr_at_k = {k: [] for k in self.ks_mrr}
        mrr_at_k_normalizer = {k: np.sum(1/np.array(list(range(1, k+1)))) for k in self.ks_mrr}
        num_seeds = set()
        for (seed_fname, cand_fname, value) in self.gt_tagging:
            if value == 0:
                continue
            num_seeds.add(seed_fname)
            seed_ind = self.image_path2query_ind[seed_fname]
            cand_ind = self.image_path2gallery_ind[cand_fname]

            for k in self.ks_mrr:
                topk_preds = top_k_inds[seed_ind]
                topk_preds = [i for i in topk_preds if self.gallery_ind2id[i] != self.query_ind2id[seed_ind]][:k]
                try:
                    rank = topk_preds.index(cand_ind) + 1.0
                    mrr_at_k[k].append(1.0 / rank)
                except:
                    mrr_at_k[k].append(0.0)

        mrr_at_k_met = {k: {} for k in self.ks_mrr}
        for k, rrs in mrr_at_k.items():
            v = np.array(rrs)
            mrr_at_k_met[k] = {'mean': (v.sum() / (len(num_seeds) * mrr_at_k_normalizer[k])), 'std': v.std()}
        return mrr_at_k_met

    def _calc_roc_pr_auc(self, top_k_scores, top_k_inds):
        scores, values = [], []
    
        for (seed_fname, cand_fname, value) in self.gt_tagging:
            seed_ind = self.image_path2query_ind[seed_fname]
            cand_ind = self.image_path2gallery_ind[cand_fname]

            topk_preds = top_k_inds[seed_ind]
            rank = topk_preds.index(cand_ind)
            score = top_k_scores[seed_ind][rank]
            values.append(value)
            scores.append(score)
        return roc_auc_score(values, scores), average_precision_score(values, scores)
            


