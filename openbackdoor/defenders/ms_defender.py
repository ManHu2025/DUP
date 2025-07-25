import os
import torch
from tqdm import tqdm
import random
from sklearn.covariance import ShrunkCovariance
from openbackdoor.victims import Victim
from openbackdoor.data import get_dataloader 
from openbackdoor.utils import logger
from typing import *
import numpy as np
import pandas as pd
import seaborn as sns
from umap import UMAP
from sklearn.metrics import roc_auc_score, calinski_harabasz_score 
from collections import defaultdict
from .defender import Defender
import matplotlib.pyplot as plt

class MSDefender(Defender):
    def __init__(
            self,
            model_name: Optional[str] = 'llama',
            data_name: Optional[str] = 'sst-2',
            attack_name: Optional[str] = 'badnets',
            frr: Optional[float] = 0.05,
            batch_size: Optional[int] = 32,
            ss_layers: Union[str, List[int]] = "all", 
            md_layers: Union[str, List[int]] = "all", 
            md_top_k: Optional[int] = 3,
            md_agg_strat: str = "mean",
            alpha: Optional[float] = 0.9,
            poison_num: Optional[int] = 200,
            visualize: bool = False, 
            purification: bool = False,
            fig_basepath: str = "./visual_results",
            **kwargs,
    ):
        super().__init__(**kwargs)
        self.model_name = model_name
        self.data_name = data_name
        self.attack_name = attack_name
        self.frr = frr
        self.batch_size = batch_size
        self.ss_layers = ss_layers 
        self.md_layers = md_layers
        self.md_agg_strat = md_agg_strat
        self.md_top_k = md_top_k
        self.alpha = alpha 
        self.poison_num = poison_num
        self.visualize = visualize
        self.purification = purification
        self.fig_basepath = fig_basepath
        self.clean_activation_means = {} 
        self.clean_activation_precisions = {}
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    @torch.no_grad()
    def get_full_representation(self, model: Victim, batch_inputs: Dict[str, torch.Tensor]):
        if self.model_name == 'bert-base-uncased':
            base_model = model.plm
            outputs = base_model(**batch_inputs, output_hidden_states=True)
            hidden_states_tuple = outputs.hidden_states
            all_layer_activations = {}
            for i in range(1, len(hidden_states_tuple)):
                layer_cls_activation = hidden_states_tuple[i][:, 0, :].detach().cpu() 
                all_layer_activations[i-1] = layer_cls_activation
        elif self.model_name in ('t5-base', 'bart-base'):
            base_model = model.plm
            outputs = base_model(**batch_inputs, output_hidden_states=True)
            encoder_hidden_states = outputs.encoder_hidden_states
            decoder_hidden_states = outputs.decoder_hidden_states
            combined_hidden_states_tuple = encoder_hidden_states[1:] + decoder_hidden_states[1:]
            all_layer_activations = {}
            for i in range(len(combined_hidden_states_tuple)):
                layer_hidden_state = combined_hidden_states_tuple[i]
                layer_activation = layer_hidden_state.mean(dim=1).detach().cpu()
                all_layer_activations[i] = layer_activation
        else: 
            base_model = model.llm.base_model

            device = next(base_model.parameters()).device
            for key, value in batch_inputs.items():
                    if isinstance(value, torch.Tensor):
                        batch_inputs[key] = value.to(device)

            outputs = base_model(**batch_inputs, output_hidden_states=True)
            hidden_states_tuple = outputs.hidden_states
            all_layer_activations = {}
            sequence_lengths = torch.sum(batch_inputs['attention_mask'], dim=1) - 1
            for i in range(1, len(hidden_states_tuple)):
                layer_hidden_state = hidden_states_tuple[i]
                last_token_activations = layer_hidden_state[torch.arange(len(layer_hidden_state)), sequence_lengths, :]
                all_layer_activations[i-1] = last_token_activations.detach().cpu()
        return {"all_layer_activations": all_layer_activations}

    def calculate_mahalanobis_distances_per_layer(self, model: Victim, batch_inputs: Dict[str, torch.Tensor], layer_indices_to_use: List[int]) -> Dict[int, torch.Tensor]:
        representations = self.get_full_representation(model, batch_inputs)
        per_layer_activations = representations["all_layer_activations"]
        per_layer_distances = {}

        for layer_idx in layer_indices_to_use:
            if layer_idx in self.clean_activation_means:
                batch_activations = per_layer_activations.get(layer_idx)
                if batch_activations is None: continue
                
                activations_tensor = batch_activations.to(self.device)
                mean_vec = self.clean_activation_means[layer_idx]
                prec_matrix = self.clean_activation_precisions[layer_idx]

                diffs = activations_tensor - mean_vec
                m_dist_sq_tensor = torch.einsum('bi,ij,bj->b', diffs, prec_matrix, diffs)
                dists = torch.sqrt(torch.relu(m_dist_sq_tensor))
                per_layer_distances[layer_idx] = dists
        return per_layer_distances

    def _extract_full_dataset_features(self, model: Victim, dataloader: torch.utils.data.DataLoader) -> np.ndarray:
        all_layer_activations_list = defaultdict(list)
        for batch in tqdm(dataloader, desc="Extracting features"):
            batch_inputs, _ = model.process(batch)
            representations = self.get_full_representation(model, batch_inputs)
            for layer_idx, batch_activations_tensor in representations["all_layer_activations"].items():
                all_layer_activations_list[layer_idx].append(batch_activations_tensor)
        
        if not all_layer_activations_list: return np.array([])
            
        num_layers = len(all_layer_activations_list)
        num_samples = sum(len(b) for b in all_layer_activations_list[0])
        hidden_size = all_layer_activations_list[0][0].shape[1]
        
        full_features = np.zeros((num_layers, num_samples, hidden_size))
        for layer_idx, tensor_list in all_layer_activations_list.items():
            full_features[layer_idx] = torch.cat(tensor_list, dim=0).numpy()
        return full_features

    def _aggregate_scores(self, per_layer_scores: Dict[int, torch.Tensor], agg_strat: str) -> torch.Tensor:
        if not per_layer_scores:
            return torch.tensor([], device=self.device)

        score_tensors = list(per_layer_scores.values())
        if not score_tensors:
            return torch.tensor([], device=self.device)
            
        stacked_scores = torch.stack(score_tensors, dim=0)

        if agg_strat == "mean":
            final_scores = torch.mean(stacked_scores, dim=0)
        else: 
            final_scores, _ = torch.max(stacked_scores, dim=0)
        return final_scores

    def calculate_spectral_scores(self, features: np.ndarray, layer_indices_to_use: List[int]) -> np.ndarray:
        if features.size == 0 or not layer_indices_to_use: return np.array([])
        
        _, num_samples, _ = features.shape
        scores = []
        for i in tqdm(range(num_samples), desc="Calculate spectral scores"):
            sample_features = features[layer_indices_to_use, i, :]
            if sample_features.shape[0] < 2: continue 
            delta_matrix = np.diff(sample_features, axis=0)
            try:
                s = np.linalg.svd(delta_matrix, compute_uv=False)
                score = s[0] / np.sum(s) if s.size > 0 and np.sum(s) > 0 else 0.0
            except np.linalg.LinAlgError:
                score = 0.0
            scores.append(score)
        return np.array(scores)
    
    def _parse_layer_selection(self, selection_strategy: Union[str, List[int]], total_layers: int) -> List[int]:
        if isinstance(selection_strategy, str):
            if selection_strategy == "first_half":
                start, end = 0, total_layers // 2
            elif selection_strategy == "middle":
                start, end = total_layers // 4, 3 * total_layers // 4
            elif selection_strategy == "last_half":
                start, end = total_layers // 2, total_layers
            elif selection_strategy == "last_quarter":
                start, end = 3 * total_layers // 4, total_layers
            elif selection_strategy == "last_three_layers":
                start, end = max(0, total_layers - 3), total_layers
            elif selection_strategy == "last_layer":
                return [total_layers - 1] if total_layers > 0 else []
            else: # "all"
                start, end = 0, total_layers
            layer_indices = list(range(start, end))
        else: 
            layer_indices = selection_strategy
        return [l for l in layer_indices if 0 <= l < total_layers]

    def plot_scores(self, md_scores, ss_scores, labels):
        fig_md, ax_md = plt.subplots(figsize=(8, 6))
        normal_md = [s for s, l in zip(md_scores, labels) if l == 0]
        anomaly_md = [s for s, l in zip(md_scores, labels) if l == 1]

        ax_md.hist(normal_md, bins='doane', label='Clean', 
                color='#0072B2', alpha=0.7, edgecolor='black', linewidth=0.5)
        ax_md.hist(anomaly_md, bins='doane', label='Poison', 
           color='#D55E00', alpha=0.7, edgecolor='black', linewidth=0.5)
        ax_md.set_title('Mahalanobis Distance (MD) Scores', fontsize=20)
        ax_md.set_xlabel('Score', fontsize=20)
        ax_md.set_ylabel('Frequency', fontsize=20)
        ax_md.tick_params(axis='both', which='major', labelsize=16)
        ax_md.legend(fontsize=20)
        ax_md.grid(True, linestyle='--', alpha=0.5)

        fig_md.tight_layout()
        md_filename = f'{self.fig_basepath}/{self.model_name}_{self.attack_name}_{self.data_name}_md_scores.pdf'
        fig_md.savefig(md_filename, dpi=300)
        plt.close(fig_md)  


        fig_ss, ax_ss = plt.subplots(figsize=(8, 6))
        normal_ss = [s for s, l in zip(ss_scores, labels) if l == 0]
        anomaly_ss = [s for s, l in zip(ss_scores, labels) if l == 1]

        ax_md.hist(normal_ss, bins='doane', label='Clean', 
                color='#0072B2', alpha=0.7, edgecolor='black', linewidth=0.5)
        ax_md.hist(anomaly_ss, bins='doane', label='Poison', 
           color='#D55E00', alpha=0.7, edgecolor='black', linewidth=0.5)
        ax_ss.set_title('Spectral Signature (SS) Scores', fontsize=20)
        ax_ss.set_xlabel('Score', fontsize=20)
        ax_ss.set_ylabel('Frequency', fontsize=20)
        ax_ss.tick_params(axis='both', which='major', labelsize=16) 
        ax_ss.legend(fontsize=20)
        ax_ss.grid(True, linestyle='--', alpha=0.5)

        fig_ss.tight_layout()
        ss_filename = f'{self.fig_basepath}/{self.model_name}_{self.attack_name}_{self.data_name}_ss_scores.pdf'
        fig_ss.savefig(ss_filename, dpi=300)
        plt.close(fig_ss) 
        
    def visualize_by_layer(
        self,
        features: np.ndarray,
        labels: List[int],
        poison_labels: List[int],
        layer_indices: List[int],
        metadata: Dict[str, str]
    ):
        logger.info('***** Visualize hidden features by layer *****')
        sns.set(style="white", rc={"axes.titlesize": 14, "axes.labelsize": 12})

        labels = np.array(labels)
        poison_labels = np.array(poison_labels, dtype=np.int64)
        num_classes = len(set(labels))
        poison_idx = np.where(poison_labels == 1)[0]

        fig_save_path = os.path.join(self.fig_basepath, f"{metadata['model']}_{metadata['attack']}_{metadata['dataset']}")
        os.makedirs(fig_save_path, exist_ok=True)
        logger.info(f"Fig save path: {fig_save_path}")

        custom_palette = ['#0072B2', '#009E73', '#F0E442', '#CC79A7']
        markers = ['o', 's', '^', 'D', 'v', '<', '>', 'P']

        for layer_idx in tqdm(layer_indices):
            layer_features = features[layer_idx]
            
            reducer = UMAP(n_components=2, random_state=42, n_neighbors=15, min_dist=0.1)
            embedding_umap = reducer.fit_transform(layer_features)
            embedding_df = pd.DataFrame(embedding_umap, columns=['UMAP-1', 'UMAP-2'])
            
            plt.figure(figsize=(8, 7), dpi=300)
            fig_title = f"{metadata['model']} on {metadata['dataset']} ({metadata['attack']}) - Layer {layer_idx}"

            for c in range(num_classes):
                class_idx = np.where(labels == c)[0]
                clean_class_idx = np.setdiff1d(class_idx, poison_idx, assume_unique=True)
                
                plt.scatter(
                    embedding_df.iloc[clean_class_idx, 0], 
                    embedding_df.iloc[clean_class_idx, 1], 
                    c=[custom_palette[c]], 
                    s=20, 
                    alpha=0.6, 
                    marker=markers[c % len(markers)],
                    label=f'Class {c} (Clean)'
                )

            plt.scatter(
                embedding_df.iloc[poison_idx, 0], 
                embedding_df.iloc[poison_idx, 1], 
                s=40, 
                c='#D55E00', 
                marker='X', 
                alpha=0.9, 
                label='Poison'
            )
            
            plt.title(fig_title, fontsize=20)
            plt.xlabel("Dimension 1", fontsize=20)
            plt.ylabel("Dimension 2", fontsize=20)
            plt.tick_params(axis='both', which='major', labelsize=16)
            plt.legend(fontsize=20, loc='best')
            plt.grid(True, linestyle='--', alpha=0.5)
            plt.tight_layout()
            
            fig_path_png = os.path.join(fig_save_path, f"layer_{layer_idx}.pdf")
            plt.savefig(fig_path_png, dpi=300, bbox_inches='tight')
            plt.close()

    def detect(self, model: Victim, clean_data: List, poison_data: List):
        model.eval()

        clean_dev_samples = clean_data["dev"][:self.poison_num]
        logger.info(f"Using {len(clean_dev_samples) // 2} to Calibrate...")
        random.seed(2025)
        random.shuffle(clean_dev_samples)
        half_dev_idx = len(clean_dev_samples) // 2
        calibration_dataset = clean_dev_samples[:half_dev_idx]
        validation_dataset = clean_dev_samples[half_dev_idx:]
        
        calibration_dataloader = get_dataloader(calibration_dataset, shuffle=False, batch_size=self.batch_size)
        validation_dataloader = get_dataloader(validation_dataset, shuffle=False, batch_size=self.batch_size)
        poison_dataloader = get_dataloader(poison_data, shuffle=False, batch_size=self.batch_size)

        validation_features = self._extract_full_dataset_features(model, validation_dataloader)
        poison_features_with_clean = self._extract_full_dataset_features(model, poison_dataloader)
        total_layers = validation_features.shape[0]

        md_layer_indices = self._parse_layer_selection(self.md_layers, total_layers)
        ss_layer_indices = self._parse_layer_selection(self.ss_layers, total_layers)

        logger.info("Calibrate MD and evaluate the weight of each layer...")
        all_layers_clean_activations_for_maha_calib = defaultdict(list)
        all_clean_labels_for_calib = []

        for batch in tqdm(calibration_dataloader, desc="Extracting calibration features"):
            batch_inputs, batch_labels = model.process(batch)
            all_clean_labels_for_calib.extend(batch_labels.cpu().numpy())
            representations_calib = self.get_full_representation(model, batch_inputs)
            for layer_idx in md_layer_indices:
                if layer_idx in representations_calib["all_layer_activations"]:
                     all_layers_clean_activations_for_maha_calib[layer_idx].append(representations_calib["all_layer_activations"][layer_idx].numpy())
        
        all_clean_labels_for_calib = np.array(all_clean_labels_for_calib)
        layer_discriminative_scores = {}

        for layer_idx in tqdm(md_layer_indices, desc="Calibrate MD and weights"):
            activations_batches_np = all_layers_clean_activations_for_maha_calib.get(layer_idx)
            if not activations_batches_np: continue
            layer_activations_np_calib = np.vstack(activations_batches_np)
            
            if layer_activations_np_calib.shape[0] < 2:
                logger.warning(f" {layer_idx} layer insufficient samples, skipping.")
                continue
            
            mean_vec = np.mean(layer_activations_np_calib, axis=0)
            self.clean_activation_means[layer_idx] = torch.from_numpy(mean_vec).float().to(self.device)
            estimator = ShrunkCovariance().fit(layer_activations_np_calib)
            self.clean_activation_precisions[layer_idx] = torch.from_numpy(estimator.precision_).float().to(self.device)

            if len(all_clean_labels_for_calib) == len(layer_activations_np_calib) and len(set(all_clean_labels_for_calib)) > 1:
                ch_score = calinski_harabasz_score(layer_activations_np_calib, all_clean_labels_for_calib)
                layer_discriminative_scores[layer_idx] = ch_score
            else:
                layer_discriminative_scores[layer_idx] = 0

        if self.md_top_k is not None and self.md_top_k > 0:
            logger.info(f"Select the top-{self.md_top_k} layers based on the weights...")
            sorted_layers = sorted(layer_discriminative_scores.items(), key=lambda item: item[1], reverse=True)
            top_k_layer_indices = [layer_idx for layer_idx, score in sorted_layers[:self.md_top_k]]
            logger.info(f"Weight of each layer: { {k: round(v, 2) for k, v in sorted_layers} }")
            logger.info(f"The final selected Top-{self.md_top_k} layer: {top_k_layer_indices}")
            md_layer_indices_final = top_k_layer_indices
        else:
            logger.info(f"No Top-K filtering is used, candidate layers: {md_layer_indices} will be used.")
            md_layer_indices_final = md_layer_indices

        md_scores_clean, ss_scores_clean = [], []
        md_scores_poison, ss_scores_poison = [], []


        logger.info("Scoring on the validation set...")
        ss_scores_clean = self.calculate_spectral_scores(validation_features, ss_layer_indices)
        for batch in tqdm(validation_dataloader):
            batch_inputs, _ = model.process(batch)
            batch_per_layer_scores = self.calculate_mahalanobis_distances_per_layer(model, batch_inputs, md_layer_indices_final)
            agg_scores = self._aggregate_scores(batch_per_layer_scores, self.md_agg_strat)
            if agg_scores.numel() > 0:
                md_scores_clean.extend(agg_scores.cpu().numpy())
        

        logger.info("Scoring on the poison set...")
        ss_scores_poison = self.calculate_spectral_scores(poison_features_with_clean, ss_layer_indices)
        for batch in tqdm(poison_dataloader):
            batch_inputs, _ = model.process(batch)
            batch_per_layer_scores = self.calculate_mahalanobis_distances_per_layer(model, batch_inputs, md_layer_indices_final)
            agg_scores = self._aggregate_scores(batch_per_layer_scores, self.md_agg_strat)
            if agg_scores.numel() > 0:
                md_scores_poison.extend(agg_scores.cpu().numpy())


        logger.info("Normalize and merge the scores...")
        md_scores_clean, md_scores_poison = np.array(md_scores_clean), np.array(md_scores_poison)
        ss_scores_clean, ss_scores_poison = np.array(ss_scores_clean), np.array(ss_scores_poison)


        if md_scores_clean.size == 0 or ss_scores_clean.size == 0:
            logger.error("The score array of clean samples is empty, and normalization and fusion cannot be performed. Please check your data and layer selection strategy.")
            return np.array([]), 0.0

        mean_md, std_md = np.mean(md_scores_clean), np.std(md_scores_clean)
        mean_ss, std_ss = np.mean(ss_scores_clean), np.std(ss_scores_clean)
        std_md = 1 if std_md < 1e-6 else std_md
        std_ss = 1 if std_ss < 1e-6 else std_ss
        
        norm_md_clean = (md_scores_clean - mean_md) / std_md
        norm_ss_clean = (ss_scores_clean - mean_ss) / std_ss
        norm_md_poison = (md_scores_poison - mean_md) / std_md
        norm_ss_poison = (ss_scores_poison - mean_ss) / std_ss
        
        logger.info(f"Using weighted fusion, the weights of the MD scores (alpha) = {self.alpha}")
        final_clean_scores = self.alpha * norm_md_clean + (1 - self.alpha) * norm_ss_clean
        final_poison_scores = self.alpha * norm_md_poison + (1 - self.alpha) * norm_ss_poison

        poison_labels = [d[2] for d in poison_data]
        original_labels = [d[1] for d in poison_data]

        os.makedirs(self.fig_basepath, exist_ok=True)
        self.plot_scores(md_scores_poison, ss_scores_poison, poison_labels)
        if self.visualize:
            if poison_features_with_clean.size > 0:
                metadata = {"dataset": self.data_name, "model": self.model_name, "attack": self.attack_name}
                self.visualize_by_layer(poison_features_with_clean, original_labels, poison_labels, ss_layer_indices, metadata)
            else:
                logger.warning("Features required for visualization not found, skipping layer visualization.")
        
        auroc = roc_auc_score(poison_labels, final_poison_scores) if len(set(poison_labels)) > 1 and len(final_poison_scores) > 0 else 0.0
        logger.info(f"AUROC: {auroc}")
        
        if self.purification:
            threshold = np.percentile(final_poison_scores, 80)
            logger.info(f"To keep the suspicious samples, the threshold is set to: {threshold}")
        else:
            threshold = np.percentile(final_clean_scores, (1 - self.frr) * 100)
            logger.info(f"Constrained FRR is {self.frr}, the threshold after fusion = {threshold}")
        
        preds = np.zeros(len(poison_data))
        if len(final_poison_scores) > 0:
            preds[final_poison_scores > threshold] = 1
        return preds, auroc
