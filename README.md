# DUP: Detection-guided Unlearning for Backdoor Purification in Language Models

This is the official repository for the paper:

> **DUP: Detection-guided Unlearning for Backdoor Purification in Language Models**

DUP introduces a unified framework that combines feature-based anomaly detection with parameter-efficient unlearning to remove backdoors from pre-trained language models. It is effective across both traditional PLMs and modern LLMs, including BERT, BART, LLaMA 3B, and Qwen 3B.

---

## Dependencies

The DUP framework is implemented and tested under the following environment:

```bash
Python: 3.10  
OS: Ubuntu 22.04  
CUDA: 11.8  
PyTorch: 2.1.2  
Accelerate: 1.0.1  
PEFT: 0.12.0  
language-tool-python: 2.8.1  
````
---

## Preparation

### Models

Download the following models and place them under the `/victim_models` directory:

* `bert-base-uncased`
* `bart-base`
* `LLaMA-3.2-3B-Instruct`
* `Qwen2.5-3B`



### Poisoned Data

The poisoned datasets are already prepared and located in `/poison_data`


---

## Attacks

To launch backdoor attacks:

```bash
python demo_attack.py
```

Supported attacks:

* BadNets
* AddSent
* StyleBackdoor
* SynBackdoor
* ...

---

## Defense: DUP Framework

### Detection Module

This module identifies poisoned inputs using Mahalanobis Distance and Spectral Signature scores with adaptive layer selection.

```bash
python demo_detection_ablation.py
```

### Purification Module

This module removes backdoors using LoRA-based distillation guided by detection:

```bash
python demo_unlearning.py
```

---

## Repository Structure

```
.
├── victim_models/               # Pretrained model weights
├── poison_data/                 # poisoned datasets
├── openbackdoor/                # Core implementation of backdoor attacks and defenses
├── demo_attack.py               # Backdoor attack scripts
├── demo_detection.py            # Detection procedure
├── demo_unlearning.py           # Unlearning procedure
└── README.md                    # Project documentation
```

