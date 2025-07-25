import torch
import torch.nn as nn
import numpy as np
import random
from tqdm import tqdm
from typing import *
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score
from scipy.stats import norm

from .defender import Defender
from openbackdoor.victims import Victim
from openbackdoor.utils import logger

def calculate_auroc(scores, labels):
    scores = [-s for s in scores]
    auroc = roc_auc_score(labels, scores)
    return auroc


def plot_score_distribution(scores, labels, title):

    normal_scores = [score for score, label in zip(scores, labels) if label == 0]
    anomaly_scores = [score for score, label in zip(scores, labels) if label == 1]

    plt.figure(figsize=(10, 6))
    plt.hist(normal_scores, bins='doane', label='Clean', color='#1f77b4', alpha=0.7, edgecolor='black')
    plt.hist(anomaly_scores, bins='doane', label='Poison', color='#ff7f0e', alpha=0.7, edgecolor='black')
    plt.xlabel('Score',fontsize=18, fontname = 'Times New Roman')
    plt.ylabel('Frequency',fontsize=18, fontname = 'Times New Roman')
    plt.title(title,fontsize=18, fontname = 'Times New Roman')
    plt.legend(fontsize=16, loc='best')
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.show()


class BadActsDefender(Defender):
    def __init__(
        self,
        frr: Optional[float] = 0.05,
        delta: Optional[float] = 2.0,
        **kwargs,
    ): 
        super().__init__(**kwargs)
        self.frr = frr
        self.delta = delta
        logger.info(f"BadActsDefender initialized with frr={self.frr}, delta={self.delta}")     

    @torch.no_grad()
    def get_attribution(self, victim: Victim, sample: str) -> List[float]:
        if hasattr(victim, 'plm'):
            base_model = victim.plm
        elif hasattr(victim, 'llm'):
            base_model = victim.llm
        else:
            base_model = victim
        
        tokenizer = victim.tokenizer
        input_tensor = tokenizer.encode(sample, add_special_tokens=True, return_tensors="pt").to(base_model.device)
        model_type = base_model.config.model_type
        outputs = base_model(input_tensor, output_hidden_states=True)
        layer_representations = []

        if model_type == 'bert':
            hidden_states = outputs.hidden_states
            layer_representations = [h[:, 0, :] for h in hidden_states]         
        elif model_type == 'bart':
            encoder_reps = [h[:, 0, :] for h in outputs.encoder_hidden_states]
            decoder_reps = [h[:, -1, :] for h in outputs.decoder_hidden_states]
            layer_representations = encoder_reps + decoder_reps
        elif model_type in ['gpt2', 'qwen2', 'llama']:
            hidden_states = outputs.hidden_states
            layer_representations = [h[:, -1, :] for h in hidden_states]
        else:
            logger.warning(f"Unsupported model type '{model_type}'. Falling back to BERT's strategy (using [CLS] token).")
            hidden_states = outputs.hidden_states
            if hidden_states is not None:
                layer_representations = [h[:, 0, :] for h in hidden_states]
            else:
                raise ValueError(f"Cannot extract hidden states for model type '{model_type}'")
        activations = []
        for i, rep in enumerate(layer_representations):
             if i > 0:
                activations.extend(rep.view(-1).cpu().numpy().tolist())
        
        return activations
    
    def _get_attributions_for_dataset(self, victim: Victim, dataset: List[Tuple[str, int, int]]) -> List[np.ndarray]:
        attributions = []
        for text, _, _ in tqdm(dataset, desc="Extracting Activations"):
            attr = self.get_attribution(victim, text)
            attributions.append(np.array(attr))
        return attributions
    
    def detect(
        self,
        model: Victim,
        clean_data: Dict[str, List],
        poison_data: List,
    ) -> Tuple[np.ndarray, float]: 
        model.eval()
        clean_dev = clean_data.get("dev", clean_data.get("validation"))
        if not clean_dev:
            raise ValueError("Clean data must contain a 'dev' or 'validation' set.")
        
        random.seed(2024) # for reproducibility
        random.shuffle(clean_dev)
        half_dev_len = len(clean_dev) // 2  

        calibration_data = clean_dev[:half_dev_len]
        threshold_data = clean_dev[half_dev_len:]   
        logger.info(f"Building statistical model on {len(calibration_data)} clean samples...")

        calibration_attributions = self._get_attributions_for_dataset(model, calibration_data)
        calibration_attributions = np.array(calibration_attributions)

        norm_params = []
        for i in range(calibration_attributions.shape[1]):
            mu, sigma = norm.fit(calibration_attributions[:, i])
            norm_params.append((mu, max(sigma, 1e-6)))

        logger.info(f"Calculating scores for {len(threshold_data)} clean samples to set threshold...")
        threshold_scores = []
        threshold_attributions = self._get_attributions_for_dataset(model, threshold_data)
        for attrs in threshold_attributions:
            in_interval = []
            for i, attr_val in enumerate(attrs):
                mu, sigma = norm_params[i]
                is_normal = int((mu - self.delta * sigma) <= attr_val <= (mu + self.delta * sigma))
                in_interval.append(is_normal)
            threshold_scores.append(np.mean(in_interval))           

        logger.info(f"Calculating scores for {len(poison_data)} target samples...")
        poison_scores = []
        poison_attributions = self._get_attributions_for_dataset(model, poison_data)
        for attrs in poison_attributions:
            in_interval = []
            for i, attr_val in enumerate(attrs):
                mu, sigma = norm_params[i]
                is_normal = int((mu - self.delta * sigma) <= attr_val <= (mu + self.delta * sigma))
                in_interval.append(is_normal)
            poison_scores.append(np.mean(in_interval))
        
        poison_labels = [d[2] for d in poison_data] 
        auroc = calculate_auroc(poison_scores, poison_labels)
        logger.info(f"Detection AUROC: {auroc:.4f}")

        # plot_score_distribution(poison_scores, poison_labels, title="Score Distribution on Target Data")

        threshold_idx = int(len(threshold_scores) * self.frr)
        threshold = np.sort(threshold_scores)[threshold_idx]
        logger.info(f"Constraining FRR to {self.frr}, threshold = {threshold:.4f}")

        preds = np.zeros(len(poison_data))
        preds[np.array(poison_scores) < threshold] = 1

        return preds, auroc
