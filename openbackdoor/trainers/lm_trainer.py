import math
from openbackdoor.victims import Victim
from openbackdoor.utils import logger, evaluate_classification
from openbackdoor.data import get_dataloader, wrap_dataset
from transformers import get_linear_schedule_with_warmup
from torch.optim import AdamW
from .trainer import Trainer
import torch
import torch.nn as nn
import os
from typing import *
from tqdm import tqdm
import numpy as np
from accelerate import Accelerator
import bitsandbytes as bnb

class LMTrainer(Trainer):
    r"""
        Trainer for language models and masked language models. Used in PLM-releasing attacks.
    """
    def __init__(
        self, 
        mlm: Optional[bool] = False,
        mlm_prob: Optional[float] = 0.15,
        batch_size: Optional[int] = 1,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.mlm = mlm
        self.mlm_prob = mlm_prob
        self.batch_size = batch_size
        self.accelerator = None 

    def set_accelerator(self, accelerator: Accelerator):
        self.accelerator = accelerator
    
    @staticmethod
    def mask_tokens(inputs, tokenizer, mlm_prob):
        labels = inputs.copy()
        probability_matrix = torch.full(labels.shape, mlm_prob)
        special_tokens_mask = [tokenizer.get_special_tokens_mask(val, already_has_special_tokens=True) for val in labels.tolist()]
        probability_matrix.masked_fill_(torch.tensor(special_tokens_mask, dtype=torch.bool), value=0.0)
        masked_indices = torch.bernoulli(probability_matrix).bool()
        labels[~masked_indices] = -1
        indices_replaced = torch.bernoulli(torch.full(labels.shape, 0.8)).bool() & masked_indices
        inputs[indices_replaced] = tokenizer.convert_tokens_to_ids(tokenizer.mask_token)
        indices_random = torch.bernoulli(torch.full(labels.shape, 0.5)).bool() & masked_indices & ~indices_replaced
        random_words = torch.randint(len(tokenizer), labels.shape, dtype=torch.long)
        inputs[indices_random] = random_words[indices_random]
        return inputs, labels
    
    def train(self, model: Victim, dataset, metrics: Optional[List[str]] = ["accuracy"]):
        dataloader = wrap_dataset(dataset, self.batch_size)
        train_dataloader = dataloader["train"]
        eval_dataloader = {key: item for key, item in dataloader.items() if key.split("-")[0] == "dev"}

        no_decay = ['bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
            {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)], 'weight_decay': self.weight_decay},
            {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
        ]
        
        self.optimizer = bnb.optim.AdamW8bit(optimizer_grouped_parameters, lr=self.lr)
        
        train_steps = len(train_dataloader) * self.epochs
        self.scheduler = get_linear_schedule_with_warmup(self.optimizer,
                                                         num_warmup_steps=self.warm_up_epochs * len(train_dataloader),
                                                         num_training_steps=train_steps)
        
        llm_to_prepare = model.llm

        self.accelerator.print(f"DEBUG: Before prepare, model to prepare is: {type(llm_to_prepare)}")
        
        # 使用 self.optimizer 和 self.scheduler 进行 prepare
        prepared_llm, self.optimizer, train_dataloader, self.scheduler = self.accelerator.prepare(
            llm_to_prepare, self.optimizer, train_dataloader, self.scheduler
        )
        model.llm = prepared_llm

        self.accelerator.print(f"DEBUG: After prepare, model is: {type(model.llm)}")

        for key in eval_dataloader:
            eval_dataloader[key] = self.accelerator.prepare(eval_dataloader[key])

        if self.accelerator.is_local_main_process:
            logger.info("***** Training with Accelerate *****")
            logger.info(f"  Num Epochs = {self.epochs}")
            logger.info(f"  Instantaneous batch size per GPU = {self.batch_size}")
            logger.info(f"  Total optimization steps = {train_steps}")

        best_dev_score = 0
        for epoch in range(self.epochs):
            avg_loss, _, _ = self.train_one_epoch(epoch, train_dataloader, model)
            if self.accelerator.is_local_main_process:
                logger.info(f'Epoch: {epoch+1}, avg loss: {avg_loss}')
            
            dev_results, dev_score = self.evaluate(model, eval_dataloader, metrics)

            if dev_score > best_dev_score:
                best_dev_score = dev_score
                if self.ckpt == 'best':
                    self.accelerator.wait_for_everyone()
                    unwrapped_model = self.accelerator.unwrap_model(model.llm)
                    if self.accelerator.is_local_main_process:
                        torch.save(unwrapped_model.state_dict(), self.model_checkpoint(self.ckpt))

        if self.ckpt == 'last':
            self.accelerator.wait_for_everyone()
            unwrapped_model = self.accelerator.unwrap_model(model.llm)
            if self.accelerator.is_local_main_process:
                torch.save(unwrapped_model.state_dict(), self.model_checkpoint(self.ckpt))

        if self.accelerator.is_local_main_process:
            logger.info("Training finished.")
        
        # 加载模型状态的操作也应在 unwrapped model 上进行
        state_dict = torch.load(self.model_checkpoint(self.ckpt), map_location='cpu')
        unwrapped_model = self.accelerator.unwrap_model(model.llm)
        unwrapped_model.load_state_dict(state_dict)
        return model


    # 修正 train_one_epoch 方法 (与上一版相同，确保所有 self.model 都已改为 model)
    def train_one_epoch(self, epoch, epoch_iterator, model):
        model.train()
        total_loss = 0
        progress_bar = tqdm(epoch_iterator, desc=f"Epoch {epoch+1} Iteration", disable=not self.accelerator.is_local_main_process)
        
        for step, batch in enumerate(progress_bar):
            batch_inputs, batch_labels = model.process(batch)
            batch_inputs = {k: v.to(self.accelerator.device) for k, v in batch_inputs.items()}
            batch_labels = batch_labels.to(self.accelerator.device)

            with self.accelerator.accumulate(model):
                if self.mlm:
                    batch_inputs['input_ids'], batch_labels = self.mask_tokens(batch_inputs['input_ids'], model.tokenizer, self.mlm_prob)
                    outputs = model(batch_inputs, masked_lm_labels=batch_labels)
                else:
                    outputs = model(batch_inputs, labels=batch_labels)
                
                loss = outputs.loss
                
                self.accelerator.backward(loss)
                
                # 关键修正 (2): 现在 self.optimizer 和 self.scheduler 是存在的
                self.optimizer.step()
                self.scheduler.step()
                self.optimizer.zero_grad()

            total_loss += loss.item()
            progress_bar.set_postfix({"loss": total_loss / (step + 1)})

        avg_loss = total_loss / len(epoch_iterator)
        # 使用 accelerator.gather 来同步所有进程的 avg_loss
        avg_loss_tensor = torch.tensor(avg_loss, device=self.accelerator.device)
        gathered_loss = self.accelerator.gather(avg_loss_tensor)
        avg_loss_all_processes = torch.mean(gathered_loss).item()
        return avg_loss_all_processes, 0, 0

    def evaluate(self, model: Victim, eval_dataloader: Dict, metrics: List[str]):
        """
        使用Accuracy作为指标，并与accelerate完全集成的评估函数。
        """
        if not hasattr(self, 'accelerator') or self.accelerator is None:
            raise RuntimeError("Accelerator not set. Please call set_accelerator before training/evaluation.")

        model.eval()  
        all_preds_dict = {key: [] for key in eval_dataloader.keys()}
        all_labels_dict = {key: [] for key in eval_dataloader.keys()}
        results = {}
        dev_scores = []
        main_metric = metrics[0] if metrics else "accuracy" # 默认为accuracy

        for key, dataloader in eval_dataloader.items():
            prepared_dataloader = self.accelerator.prepare(dataloader)
            
            logger.info(f"***** Running evaluation on {key} *****")

            for batch in tqdm(prepared_dataloader, desc=f"Evaluating {key}", disable=not self.accelerator.is_local_main_process):
                batch_inputs, batch_labels = model.process(batch)
                
                with torch.no_grad():
                    outputs = model(batch_inputs)
                    preds = torch.argmax(outputs.logits, dim=-1)

                gathered_preds, gathered_labels = self.accelerator.gather_for_metrics((preds, batch_labels))
                
                all_preds_dict[key].append(gathered_preds.cpu())
                all_labels_dict[key].append(gathered_labels.cpu())

        if self.accelerator.is_main_process:
            for key in eval_dataloader.keys():
                all_preds = torch.cat(all_preds_dict[key])
                all_labels = torch.cat(all_labels_dict[key])
                
                accuracy = (all_preds == all_labels).float().mean().item()
                logger.info(f"  Accuracy on {key}: {accuracy}")
                if key not in results:
                    results[key] = {}
                results[key]["accuracy"] = accuracy

                if "dev" in key and main_metric == "accuracy":
                    dev_scores.append(accuracy)

        final_score = np.mean(dev_scores) if self.accelerator.is_main_process and dev_scores else 0.0
        
        return results, final_score