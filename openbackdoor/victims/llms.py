from typing import *
from .victim import Victim
import torch
from transformers import AutoConfig, AutoTokenizer, AutoModelForSequenceClassification

class LLMVictim(Victim):

    def __init__(
        self, 
        model: Optional[str] = "Qwen2.5-0.5B-Instruct",
        path: Optional[str] = "./victim_models/Qwen2.5-0.5B-Instruct",
        num_classes: Optional[int] = 2,
        max_len: Optional[int] = 512,
        **kwargs
    ):
        super().__init__()

        self.model_name = model
        self.model_config = AutoConfig.from_pretrained(path)
        self.model_config.num_labels = num_classes

        self.llm = AutoModelForSequenceClassification.from_pretrained(path, config=self.model_config)
        self.tokenizer = AutoTokenizer.from_pretrained(path)
        
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        self.llm.config.pad_token_id =  self.tokenizer.eos_token_id
        self.llm.gradient_checkpointing_enable()
        self.max_len = max_len

    def to(self, device):
        self.llm = self.llm.to(device)
    
    def forward(self, inputs, labels=None):
        device = next(self.llm.parameters()).device
        for key, value in inputs.items():
            if isinstance(value, torch.Tensor):
                inputs[key] = value.to(device)

        if labels is not None:
            labels = labels.to(device)

        return self.llm(**inputs, labels=labels, output_hidden_states=True, return_dict=True)
    
    def process(self, batch):
        text = batch["text"]
        labels = batch["label"]
        input_batch = self.tokenizer(text, padding=True, truncation=True, max_length=self.max_len, return_tensors="pt")
        return input_batch, labels
    
    def get_repr_embeddings(self, inputs):
        output = getattr(self.llm, self.model_name)(**inputs).last_hidden_state

        return output[:, 0, :]
    
    @property
    def word_embedding(self):
        head_name = [n for n, c in self.llm.named_children()][0]
        layer = getattr(self.llm, head_name)
        return layer.embeddings.word_embeddings.weight