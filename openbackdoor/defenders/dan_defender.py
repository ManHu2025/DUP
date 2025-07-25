from .defender import Defender
from tqdm import tqdm
from sklearn.covariance import ShrunkCovariance
from openbackdoor.victims import Victim
from openbackdoor.data import get_dataloader, collate_fn
from openbackdoor.utils import logger
from typing import *
from sklearn.feature_extraction.text import TfidfVectorizer
from torch.utils.data import DataLoader
import random
import numpy as np
import torch
from torch.nn import CosineSimilarity
import torch.nn.functional as F
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score

def calculate_auroc(scores, labels):
    scores = [-s for s in scores]
    auroc = roc_auc_score(labels, scores)
    return auroc

def plot_score_distribution(scores, labels, targert):
    normal_scores = [score for score, label in zip(scores, labels) if label == 0]
    anomaly_scores = [score for score, label in zip(scores, labels) if label == 1]
    plt.figure(figsize=(8, 6))
    plt.hist(normal_scores, bins='doane', label='Clean', alpha=.8, edgecolor='black')
    plt.hist(anomaly_scores, bins='doane', label='Poison', alpha=.8, edgecolor='black')
    plt.xlabel('Score')
    plt.ylabel('Frequency')
    plt.title('Score Distribution')
    plt.legend()
    plt.show()


class DANDefender(Defender):

    def __init__(
            self,
            model_name: Optional[str] = 'llama',
            batch_size: Optional[int] = 4, 
            frr: Optional[float] = 0.05,
            poison_dataset: Optional[str] = 'sst-2',
            attacker: Optional[str] = 'badnets',
            **kwargs,
    ):
        super().__init__(**kwargs)
        self.model_name = model_name
        self.batch_size = batch_size
        self.frr = frr
        self.poison_dataset = poison_dataset
        self.attacker = attacker

    def _get_embeddings_for_batch(self, text_batch: List[str]):
        with torch.no_grad():
            batch = self.tokenizer(text_batch, padding=True, truncation=True, return_tensors="pt", max_length=512).to(self.model.device)
            outputs = self.model(**batch, output_hidden_states=True)
            try:
                hidden_states = outputs.hidden_states
                hidden_states = torch.cat([h.unsqueeze(0) for h in hidden_states], dim=0)
            except Exception:  # for BART models
                encoder_hidden_states = outputs.encoder_hidden_states
                decoder_hidden_states = outputs.decoder_hidden_states
                hidden_states = torch.cat([h.unsqueeze(0) for h in encoder_hidden_states] + [h.unsqueeze(0) for h in decoder_hidden_states][1:], dim=0)
            
            attention_masks = batch['attention_mask']
            input_mask_expanded = attention_masks.unsqueeze(-1).expand(hidden_states.size()).float()
            sum_embeddings = torch.sum(hidden_states * input_mask_expanded, 2)
            sum_mask = torch.clamp(input_mask_expanded.sum(2), min=1e-9)
            avg_hidden_states = sum_embeddings / sum_mask
            
            del outputs, batch, hidden_states, attention_masks, input_mask_expanded, sum_embeddings, sum_mask
            torch.cuda.empty_cache()

            return avg_hidden_states.cpu().numpy()

    def detect(
            self,
            model: Victim,
            clean_data: List,
            poison_data: List,
    ):
        distance_metric = 'maha'
        
        if self.model_name in ('bert-base-uncased', 'bart-base'):
            self.model = model.plm
            self.tokenizer = model.tokenizer
        elif self.model_name in ('Qwen2.5-0.5B-Instruct', 'Qwen2.5-3B', 'Llama-3.2-3B-Instruct'):
            self.model = model.llm.base_model
            self.tokenizer = model.tokenizer
        self.model.eval()

        logger.info("Phase 1: Calculating statistics from clean dev data...")
        clean_dev = clean_data["dev"]
        clean_dev_texts = [d[0] for d in clean_dev]
        clean_dev_labels = np.array([d[1] for d in clean_dev])
        
        all_clean_features_list = []
        total_batches = (len(clean_dev_texts) + self.batch_size - 1) // self.batch_size
        
        for i in tqdm(range(total_batches), desc="Processing clean data"):
            batch_texts = clean_dev_texts[i * self.batch_size: (i + 1) * self.batch_size]
            batch_features = self._get_embeddings_for_batch(batch_texts)
            all_clean_features_list.append(batch_features)

        all_clean_features = np.concatenate(all_clean_features_list, axis=1)
        del all_clean_features_list
        
        num_layers = all_clean_features.shape[0] - 1
        indices = np.arange(all_clean_features.shape[1])
        np.random.seed(2024)
        np.random.shuffle(indices)
        valid_size = int(0.2 * all_clean_features.shape[1])
        
        train_indices = indices[:-valid_size]
        valid_indices = indices[-valid_size:]
        
        clean_features_train = all_clean_features[:, train_indices]
        clean_labels_train = clean_dev_labels[train_indices]
        clean_features_valid = all_clean_features[:, valid_indices]

        valid_scores_list = []
        estimators = [] 
        for layer in range(1, num_layers + 1):
            sample_class_mean, precision = self.sample_estimator(clean_features_train[layer], clean_labels_train)
            estimators.append({'mean': sample_class_mean, 'precision': precision})
            
            valid_scores_for_layer = -1 * self.get_distance_score(sample_class_mean, precision, clean_features_valid[layer], measure=distance_metric)
            mean, std = np.mean(valid_scores_for_layer), np.std(valid_scores_for_layer)
            valid_scores_list.append((-1 * (valid_scores_for_layer - mean) / std))
        
        del all_clean_features, clean_features_train, clean_features_valid
        
        valid_scores = np.mean(valid_scores_list, axis=0)
        threshold = np.nanpercentile(valid_scores, self.frr * 100)
        logger.info(f"Constrain FRR to {self.frr}, threshold = {threshold:.4f}")

        logger.info("Phase 2: Scoring poison data...")
        poison_texts = [d[0] for d in poison_data]
        poison_labels = [d[2] for d in poison_data]
        
        final_poison_scores_list = []
        total_poison_batches = (len(poison_texts) + self.batch_size - 1) // self.batch_size

        for i in tqdm(range(total_poison_batches), desc="Processing poison data"):
            batch_texts = poison_texts[i * self.batch_size: (i + 1) * self.batch_size]
            batch_features = self._get_embeddings_for_batch(batch_texts)
            
            batch_scores_list = []
            for layer in range(1, num_layers + 1):
                estimator = estimators[layer-1] # a dict of {'mean': ..., 'precision': ...}
                layer_scores = -1 * self.get_distance_score(estimator['mean'], estimator['precision'], batch_features[layer], measure=distance_metric)
                mean = np.mean(valid_scores_list[layer-1]) 
                std = np.std(valid_scores_list[layer-1])
                layer_scores = (layer_scores - mean) / std
                batch_scores_list.append(-1 * layer_scores)
            
            batch_scores = np.mean(batch_scores_list, axis=0)
            final_poison_scores_list.append(batch_scores)

        final_poison_scores = np.concatenate(final_poison_scores_list)
        
        auroc = calculate_auroc(final_poison_scores, poison_labels)
        logger.info(f"AUROC: {auroc:.4f}")
        
        preds = np.zeros(len(poison_data))
        preds[final_poison_scores < threshold] = 1
        
        return preds, auroc
    def sample_estimator(self,features, labels):
        labels = labels.reshape(-1)
        num_classes = np.unique(labels).shape[0]
        group_lasso = ShrunkCovariance()
        sample_class_mean = []
        for c in range(num_classes):
            current_class_mean = np.mean(features[labels==c,:], axis=0)
            sample_class_mean.append(current_class_mean)
        X = [features[labels==c,:] - sample_class_mean[c] for c in range(num_classes)]
        X = np.concatenate(X, axis=0)
        group_lasso.fit(X)
        precision = group_lasso.precision_
        return sample_class_mean, precision

    def get_distance_score(self,class_mean, precision, features, measure='maha'):
        num_classes = len(class_mean)
        num_samples = len(features)
        class_mean = [torch.from_numpy(m).float() for m in class_mean]
        precision = torch.from_numpy(precision).float()
        features = torch.from_numpy(features).float()
        scores = []
        for c in range(num_classes):
            centered_features = features.data - class_mean[c]
            if measure == 'maha':
                score = -1.0*torch.mm(torch.mm(centered_features, precision), centered_features.t()).diag()
            elif measure == 'euclid':
                score = -1.0*torch.mm(centered_features, centered_features.t()).diag()
            elif measure == 'cosine':
                score = torch.tensor([CosineSimilarity()(features[i].reshape(1,-1), class_mean[c].reshape(1,-1)) for i in range(num_samples)])
            scores.append(score.reshape(-1,1))
        scores = torch.cat(scores, dim=1)
        scores,_ = torch.max(scores, dim=1)
        scores = scores.cpu().numpy()
        return scores







# class DANDefender(Defender):

#     def __init__(
#             self,
#             model_name: Optional[str] = 'llama',
#             batch_size: Optional[int] = 32,
#             frr: Optional[float] = 0.05,
#             poison_dataset: Optional[str] = 'sst-2',
#             attacker: Optional[str] = 'badnets',
#             **kwargs,
#     ):
#         super().__init__(**kwargs)
#         self.model_name = model_name
#         self.batch_size = batch_size
#         self.frr = frr
#         self.poison_dataset = poison_dataset
#         self.attacker = attacker

#     def detect(
#             self,
#             model: Victim,
#             clean_data: List,
#             poison_data: List,
#     ):
#         distance_metric ='maha'
#         std = True
#         agg = 'mean'

#         clean_dev = clean_data["dev"]
#         if self.model_name in ('bert-base-uncased', 'bart-base'):
#             self.model = model.plm
#             self.tokenizer = model.tokenizer
#         elif self.model_name in ('Qwen2.5-0.5B-Instruct', 'Qwen2.5-3B', 'Llama-3.2-3B-Instruct'):
#             self.model = model.llm.base_model
#             self.tokenizer = model.tokenizer
#         self.model.eval()
#         # self.target_label = self.get_target_label(poison_data)
#         clean_dev_texts = [d[0] for d in clean_dev]
#         clean_dev_labels = np.array([d[1] for d in clean_dev])
#         clean_dev_feature = self.get_embeddings(self.model,self.tokenizer,clean_dev_texts,self.batch_size,self.model.device)

#         poison_texts = [d[0] for d in poison_data]
#         poison_feature = self.get_embeddings(self.model, self.tokenizer, poison_texts, self.batch_size,
#                                                 self.model.device)


#         ind_dev_features = clean_dev_feature
#         # ind_train_features = torch.load('{}/{}_ind_train_features.pt'.format(input_dir, token_pooling))
#         ind_dev_labels = clean_dev_labels

#         num_layers = ind_dev_features.shape[0] - 1
#         indices = np.arange(ind_dev_features.shape[1])
#         np.random.seed(2024)
#         np.random.shuffle(indices)
#         valid_size = int(0.2 * ind_dev_features.shape[1])
#         ind_dev_features_train, ind_dev_features_valid = ind_dev_features[:, indices[:-valid_size]], ind_dev_features[:,
#                                                                                                      indices[
#                                                                                                      -valid_size:]]
#         ind_dev_labels_train, ind_dev_labels_valid = ind_dev_labels[indices[:-valid_size]], ind_dev_labels[
#             indices[-valid_size:]]

#         poison_test_features = poison_feature

#         poison_scores_list = []
#         valid_scores_list = []
#         for layer in range(1, num_layers + 1):
#             ind_train_features = ind_dev_features_train[layer]
#             sample_class_mean, precision = self.sample_estimator(ind_train_features, ind_dev_labels_train)
#             valid_scores = -1 * self.get_distance_score(sample_class_mean, precision, ind_dev_features_valid[layer],
#                                                    measure=distance_metric)
#             poison_scores = -1 * self.get_distance_score(sample_class_mean, precision, poison_test_features[layer],
#                                                     measure=distance_metric)
#             if std:
#                 mean = np.mean(valid_scores)
#                 std = np.std(valid_scores)
#                 valid_scores = (valid_scores - mean) / std
#                 poison_scores = (poison_scores - mean) / std
#             poison_scores_list.append(-1 * poison_scores)
#             valid_scores_list.append(-1 * valid_scores)
#         if agg == 'mean':
#             valid_scores = np.mean(valid_scores_list, axis=0)
#             poison_scores = np.mean(poison_scores_list, axis=0)
#         elif agg == 'min':
#             valid_scores = np.min(valid_scores_list, axis=0)
#             poison_scores = np.min(poison_scores_list, axis=0)

#         poison_labels = [d[2] for d in poison_data]
#         auroc = calculate_auroc(poison_scores,poison_labels)
#         logger.info("auroc: {}".format(auroc))
#         plot_score_distribution(poison_scores,poison_labels,self.poison_dataset+'-'+self.attacker)

#         threshold = np.nanpercentile(valid_scores, self.frr * 100)
#         logger.info("Constrain FRR to {}, threshold = {}".format(self.frr, threshold))
#         preds = np.zeros(len(poison_data))
#         # poisoned_idx = np.where(poison_prob < threshold)
#         # logger.info(poisoned_idx.shape)
#         preds[poison_scores < threshold] = 1

#         return preds,auroc

#     def get_embeddings(self, model, tokenizer, text_list, batch_size, device, target_label=None):

#         total_eval_len = len(text_list)
#         if total_eval_len % batch_size == 0:
#             NUM_EVAL_ITER = int(total_eval_len / batch_size)
#         else:
#             NUM_EVAL_ITER = int(total_eval_len / batch_size) + 1

#         # cls_features = []
#         # max_features = []
#         avg_features = []
#         with torch.no_grad():
#             for i in tqdm(range(NUM_EVAL_ITER)):
#                 batch_sentences = text_list[i * batch_size: min((i + 1) * batch_size, total_eval_len)]
#                 batch = tokenizer(batch_sentences, padding=True, truncation=True, return_tensors="pt",
#                                   max_length=512).to(device)
#                 outputs = model(**batch, output_hidden_states=True)
#                 try:
#                     hidden_states = outputs.hidden_states
#                     hidden_states = torch.cat([h.unsqueeze(0) for h in hidden_states],
#                                               dim=0)  # layers, batch_size, sequence_length, hidden_size
#                 except Exception:  # bart
#                     encoder_hidden_states = outputs.encoder_hidden_states
#                     decoder_hidden_states = outputs.decoder_hidden_states
#                     hidden_states = torch.cat([h.unsqueeze(0) for h in encoder_hidden_states] + [h.unsqueeze(0) for h in
#                                                                                                  decoder_hidden_states][
#                                                                                                 1:], dim=0)
#                 # cls_hidden_states = hidden_states[:, :, 0, :]
#                 attention_masks = batch['attention_mask']
#                 # input_mask_expanded = attention_masks.unsqueeze(-1).expand(hidden_states.size()).float()
#                 # max_hidden_states = hidden_states
#                 # max_hidden_states[input_mask_expanded == 0] = -1e9  # Set padding tokens to large negative value
#                 # max_hidden_states = torch.max(hidden_states, 2)[0]
#                 input_mask_expanded = attention_masks.unsqueeze(-1).expand(hidden_states.size()).float()
#                 sum_embeddings = torch.sum(hidden_states * input_mask_expanded, 2)
#                 sum_mask = input_mask_expanded.sum(2)
#                 sum_mask = torch.clamp(sum_mask, min=1e-9)
#                 avg_hidden_states = sum_embeddings / sum_mask

#                 if target_label is not None:
#                     logits = outputs.logits
#                     predict_labels = np.array(torch.argmax(logits, dim=1).cpu())
#                     indices = np.argwhere(predict_labels == target_label)
#                     # cls_features.append(np.squeeze(cls_hidden_states.cpu().numpy()[:, indices, :]))
#                     # max_features.append(np.squeeze(max_hidden_states.cpu().numpy()[:, indices, :]))
#                     avg_features.append(np.squeeze(avg_hidden_states.cpu().numpy()[:, indices, :]))
#                 else:
#                     # cls_features.append(cls_hidden_states.cpu().numpy())
#                     # max_features.append(max_hidden_states.cpu().numpy())
#                     avg_features.append(avg_hidden_states.cpu().numpy())
#             # cls_features = np.concatenate(cls_features, axis=1)
#             # max_features = np.concatenate(max_features, axis=1)
#             avg_features = np.concatenate(avg_features, axis=1)

#             del hidden_states, outputs, batch, avg_hidden_states, input_mask_expanded
#             torch.cuda.empty_cache() 
#         # print(cls_features.shape)
#         return avg_features

#     def pooling_features(self,features, pooling='last', fusion_module=None): # layers, num_samples, hidden_size
#         num_layers = features.shape[0]
#         if pooling == 'last':
#             return features[-1,:,:]
#         elif pooling == 'avg':
#             return np.mean(features[1:], axis=0)
#         elif pooling == 'avg_emb': # including token embeddings
#             return np.mean(features, axis=0)
#         elif pooling == 'emb':
#             return features[0]
#         elif pooling == 'first_last':
#             return (features[-1] + features[1])/2.0
#         elif pooling == 'odd':
#             odd_layers= [1 + i for i in range(0, num_layers-1,2)]
#             return (np.sum(features[odd_layers],axis=0))/(num_layers/2)
#         elif pooling == 'even':
#             even_layers= [2 + i for i in range(0, num_layers-1,2)]
#             return (np.sum(features[even_layers],axis=0))/(num_layers/2)
#         elif pooling == 'last2':
#             return (features[-1] + features[-2])/2.0
#         elif pooling == 'concat':
#             features =  np.transpose(features, (1,0,2)) # num_samples, layers, hidden_size
#             return features.reshape(features.shape[0],-1) # num_samples, layers*hidden_size
#         elif type(pooling) == int or (type(pooling) == str and pooling.isdigit()):
#             pooling = int(pooling)
#             return features[pooling]
#         elif ',' in pooling or type(pooling) == list:
#             layers = pooling
#             if type(pooling) == str:
#                 layers = list([int(l) for l in pooling.split(',')])
#             return np.mean(features[layers], axis=0)
#         else:
#             raise NotImplementedError

#     def sample_estimator(self,features, labels):
#         labels = labels.reshape(-1)
#         num_classes = np.unique(labels).shape[0]
#         # print(num_classes)
#         #group_lasso = EmpiricalCovariance(assume_centered=False)
#         #group_lasso =  MinCovDet(assume_centered=False, random_state=42, support_fraction=1.0)
#         group_lasso = ShrunkCovariance()
#         sample_class_mean = []
#         #class_covs = []
#         for c in range(num_classes):
#             current_class_mean = np.mean(features[labels==c,:], axis=0)
#             sample_class_mean.append(current_class_mean)
#             #cov_now = np.cov((features[labels == c]-(current_class_mean.reshape([1,-1]))).T)
#             #class_covs.append(cov_now)
#         #precision = np.linalg.inv(np.mean(np.stack(class_covs,axis=0),axis=0))
#         #print(precision.shape)
#         #
#         X  = [features[labels==c,:] - sample_class_mean[c]  for c in range(num_classes)]
#         X = np.concatenate(X, axis=0)
#         group_lasso.fit(X)
#         precision = group_lasso.precision_

#         return sample_class_mean, precision

#     def get_distance_score(self,class_mean, precision, features, measure='maha'):
#         num_classes = len(class_mean)
#         num_samples = len(features)
#         class_mean = [torch.from_numpy(m).float() for m in class_mean]
#         precision = torch.from_numpy(precision).float()
#         features = torch.from_numpy(features).float()
#         scores = []
#         for c in range(num_classes):
#             centered_features = features.data - class_mean[c]
#             if measure == 'maha':
#                 score = -1.0*torch.mm(torch.mm(centered_features, precision), centered_features.t()).diag()
#             elif measure == 'euclid':
#                 score = -1.0*torch.mm(centered_features, centered_features.t()).diag()
#             elif measure == 'cosine':
#                 score = torch.tensor([CosineSimilarity()(features[i].reshape(1,-1), class_mean[c].reshape(1,-1)) for i in range(num_samples)])
#             scores.append(score.reshape(-1,1))
#         scores = torch.cat(scores, dim=1) # num_samples, num_classes
#         # print(scores.shape)
#         scores,_ = torch.max(scores, dim=1) # num_samples
#         #scores = scores[:,1]
#         scores = scores.cpu().numpy()
#         return scores
