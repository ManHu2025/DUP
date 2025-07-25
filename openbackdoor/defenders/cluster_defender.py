import os
import torch
from typing import *
from tqdm import tqdm
import pandas as pd
import numpy as np
from umap import UMAP
from sklearn.decomposition import PCA
from hdbscan import HDBSCAN
import seaborn as sns
from matplotlib import pyplot as plt
from .defender import Defender
from openbackdoor.victims import Victim
from openbackdoor.utils import logger
from openbackdoor.data import get_dataloader
from sklearn.mixture import BayesianGaussianMixture
from sklearn.metrics import pairwise_distances



class ClusterDefender(Defender):
    def __init__(
        self, 
        batch_size = 16,
        visual = True,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.pre = False
        self.correct = False
        self.batch_size = batch_size
        self.visual = visual
    
    def detect(
        self, 
        model: Victim, 
        clean_data: List, 
        poison_data: List
    ):
        # clean data包含train、eval和test三个干净数据
        # poison data为test clean和poison的结合
        # 提取受害者模型的隐藏层特征，进行聚类
        # 聚类密度更高的簇可能是中毒数据
        num_classes = len(set([d[1] for d in poison_data]))
        poison_dataloader = get_dataloader(poison_data, shuffle=False, batch_size=self.batch_size)

        hidden_states, true_labels, poison_labels = self.get_hidden_state(model, poison_dataloader)

        embeddings = self.dimension_reduction(hidden_states, 
                                              pca_components=20, 
                                              n_neighbors=100, 
                                              min_dist=0.5, 
                                              umap_conponents=2)

        if self.visual:
            self.visualization(embeddings, true_labels, poison_labels)

        pred_labels = self.clustering(embeddings, n_components=num_classes+1, random_state=42)

        if self.visual:
            self.visual_cluster(embeddings, pred_labels)

        # 根据簇内距离得到投毒簇。
        pison_label = self.detect_poison_class(embeddings, pred_labels)


        # pred_labels = np.where(cluster_labels==-1, 1, 0)
        
        # 使用定量指标评估预测标签结果。（多少是正确的，多少是误判的）FAR、FRR、

        # filtered_dataset = self.filtering(poison_data, true_labels, pred_labels)

        auc = 0

        return pred_labels, auc

    def get_hidden_state(self, model, dataloader):
        model.eval()
        hidden_states = []
        labels = []
        poison_labels = []
        logger.info('***** Computing hidden hidden_state *****')
        for batch in tqdm(dataloader):
            text, label, poison_label = batch['text'], batch['label'], batch['poison_label']
            labels.extend(label)
            poison_labels.extend(poison_label)
            batch_inputs, _ = model.process(batch)
            output = model(batch_inputs)
            # Only take the hidden state of the last layer
            hidden_state = output.hidden_states[-1]
            if "attention_mask" in batch_inputs:
                sequence_lengths = torch.sum(batch_inputs["attention_mask"], dim=1) - 1
                batch_indices = torch.arange(hidden_state.shape[0])
                pooled_output = hidden_state[batch_indices, sequence_lengths, :]
            else:
                pooled_output = hidden_state[:, -1, :]
            hidden_states.extend(pooled_output.detach().cpu().tolist())
        return hidden_states, labels, poison_labels

    def dimension_reduction(self, 
        hidden_states, pca_components=20, 
        n_neighbors=100, min_dist=0.5, 
        umap_conponents=2
    ):
        logger.info('***** Dimension Reduction *****')
        pca = PCA(n_components=pca_components)

        umap = UMAP(n_neighbors=n_neighbors,
                    min_dist=min_dist,
                    n_components=umap_conponents,
                    transform_seed=42)
        
        embedding_pca = pca.fit_transform(hidden_states)
        embedding_umap = umap.fit(embedding_pca).embedding_

        return embedding_umap
    
    def clustering(self, embeddings,
                   n_components,
                   random_state):
        
        logger.info('***** Clustering *****')
        bgm = BayesianGaussianMixture(n_components=n_components, random_state=random_state)  # 实际簇数可能小于n_components
        pred_labels = bgm.fit_predict(embeddings)

        return pred_labels
    
    def filtering(self, dataset: List, y_true: List, y_pred: List):
        
        logger.info("Filtering suspicious samples")

        dropped_indices = []
        if isinstance(y_true[0], torch.Tensor):
            y_true = [y.item() for y in y_true]

        for true_label in set(y_true):
            
            # 找到所有等于true label的索引
            groundtruth_samples = np.where(y_true==true_label*np.ones_like(y_true))[0]
            
            drop_scale = 0.5*len(groundtruth_samples)

            # Check the predictions for samples of this groundtruth label
            predictions = set()
            for i, pred in enumerate(y_pred):
                if i in groundtruth_samples:
                    predictions.add(pred)

            if len(predictions) > 1:
                count = pd.DataFrame(columns=['predictions'])

                for pred_label in predictions:
                    count.loc[pred_label,'predictions'] = \
                        np.sum(np.where((y_true==true_label*np.ones_like(y_true))*\
                                        (y_pred==pred_label*np.ones_like(y_pred)), 
                                    np.ones_like(y_pred), np.zeros_like(y_pred)))
                cluster_order = count.sort_values(by='predictions', ascending=True)
                
                # we always preserve the largest prediction cluster
                for pred_label in cluster_order.index.values[:-1]: 
                    item = cluster_order.loc[pred_label, 'predictions']
                    if item < drop_scale:

                        idx = np.where((y_true==true_label*np.ones_like(y_true))*\
                                        (y_pred==pred_label*np.ones_like(y_pred)))[0].tolist()

                        dropped_indices.extend(idx)

        filtered_dataset = []
        for i, data in enumerate(dataset):
            if i not in dropped_indices:
                filtered_dataset.append(data)
        
        return filtered_dataset

    def detect_poison_class(self, embeddings, labels):
        unique_labels = np.unique(labels)
        compactness = {}
        
        for label in unique_labels:
            if label == -1:  # 忽略噪声点（如果有）
                continue
            class_samples = embeddings[labels == label]
            
            # 计算类内平均距离
            intra_distances = pairwise_distances(class_samples)
            avg_distance = np.mean(intra_distances)
            
            # 计算类内标准差（可选）
            centroid = np.mean(class_samples, axis=0)
            distances_to_centroid = np.linalg.norm(class_samples - centroid, axis=1)
            std_distance = np.std(distances_to_centroid)
            
            compactness[label] = {
                'avg_intra_distance': avg_distance,
                'std_distance': std_distance
            }
        print(compactness)
        # 选择最紧密的类作为中毒候选
        poison_label = min(compactness.items(), key=lambda x: x[1]['avg_intra_distance'])[0]
        return poison_label

    def visualization(self, embedding_umap, labels,
                      poison_labels, fig_basepath="./vis_unlearn"):
        embeddings = pd.DataFrame(embedding_umap)
        labels = np.array(labels)
        poison_idx = np.where(poison_labels==np.ones_like(poison_labels))[0]
        num_classes = len(set(labels))

        sns.set(style="white", rc={"axes.titlesize":18, "axes.labelsize":16})
        markers = ['o', 's', 'D', '^', 'v', '<', '>', 'P']
        palette = sns.color_palette("Set2", num_classes)

        plt.figure(figsize=(7, 6), dpi=300)
        for c in range(num_classes):
            idx = np.where(labels==int(c)*np.ones_like(labels))[0]
            idx = list(set(idx) ^ set(poison_idx))
            plt.scatter(embeddings.iloc[idx, 0], embeddings.iloc[idx, 1], c=palette[c], s=18, alpha=0.7, marker=markers[c % len(markers)], label=f'Class {c}')
        
        plt.scatter(embeddings.iloc[poison_idx, 0], embeddings.iloc[poison_idx, 1], s=30, c='gray', marker='X', alpha=0.7, label='Poison')

        plt.tick_params(labelsize='large')
        plt.legend(fontsize=13, loc='best', frameon=True)
        plt.tight_layout()
        os.makedirs(fig_basepath, exist_ok=True)
        plt.savefig(os.path.join(fig_basepath, f'test_data.png'), dpi=300, bbox_inches='tight')
        plt.savefig(os.path.join(fig_basepath, f'test_data.pdf'), dpi=300, bbox_inches='tight')
        fig_path = os.path.join(fig_basepath, f'test_data.png')
        logger.info(f'Saving png to {fig_path}')
        plt.close()

    def visual_cluster(self, embeddings, pred_labels):
        plt.figure(figsize=(10, 6))
        scatter = plt.scatter(
            embeddings[:, 0],  # x 轴
            embeddings[:, 1],  # y 轴
            c=pred_labels,    # 颜色对应聚类标签
            cmap='Spectral',   # 颜色映射（适合区分多类别）
            s=5,              # 点大小
            alpha=0.8         # 透明度
        )
        plt.colorbar(scatter, label='Cluster Label')
        plt.title('BayesianGaussianMixture Clustering (2D)')
        plt.xlabel('Dimension 1')
        plt.ylabel('Dimension 2')
        plt.savefig(os.path.join("./vis_unlearn", f'BayesianGaussianMixture.png'), dpi=300, bbox_inches='tight')
        plt.close()
