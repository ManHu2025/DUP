import json
import os
import wandb
import random
import numpy as np
from openbackdoor.defenders import load_defender
import torch
import pandas as pd
import types
import gc
from openbackdoor.attackers import load_attacker
from openbackdoor.data import load_dataset, get_dataloader
from openbackdoor.victims import load_victim
from openbackdoor.utils import set_config, logger
from openbackdoor.utils.visualize import display_results
from torch.nn import KLDivLoss, LogSoftmax, Softmax
from peft import get_peft_model, LoraConfig, TaskType
from transformers import get_scheduler
from torch.nn import CrossEntropyLoss

os.environ["WANDB_API_KEY"] = ''
os.environ["WANDB_MODE"] = "offline"

def seed_everything(seed=2025):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def load_poison_data(path):
    if path is not None:
        data = pd.read_csv(os.path.join(path)).values
        poisoned_data = [(d[1], d[2], d[3]) for d in data]
        return poisoned_data

def pad_to_max_length(tensor, max_len, pad_value):
    batch_size, seq_len = tensor.size()
    if seq_len == max_len:
        return tensor
    padding_len = max_len - seq_len
    padding = torch.full((batch_size, padding_len), pad_value, dtype=tensor.dtype, device=tensor.device)
    return torch.cat([tensor, padding], dim=1)

def universal_forward(
    self,
    batch_or_input_ids=None, 
    attention_mask=None,
    token_type_ids=None,
    labels=None, 
    **kwargs,
):
    input_dict = None
    
    if isinstance(batch_or_input_ids, dict):
        input_dict = batch_or_input_ids
    else:
        input_dict = {
            "input_ids": batch_or_input_ids,
            "attention_mask": attention_mask,
            "token_type_ids": token_type_ids,
            **kwargs
        }

    if hasattr(self, 'plm'):
        base_model = self.plm
    elif hasattr(self, 'llm'):
        base_model = self.llm
    else:
        raise AttributeError("Victim model has neither 'plm' nor 'llm' attribute.")

    final_inputs = {k: v for k, v in input_dict.items() if v is not None}
    output = base_model(**final_inputs)
    return {"logits": output.logits}

def eval_asr(model, dataloader, target_label, device):
    model.eval()
    total = 0
    hit = 0
    with torch.no_grad():
        for batch in dataloader:
            inputs, labels = model.module.process(batch) if hasattr(model, 'module') else model.process(batch)
    
            for key in inputs:
                if isinstance(inputs[key], torch.Tensor):
                    inputs[key] = inputs[key].to(device)
    
            logits = model(**inputs)['logits'] if hasattr(model, '__call__') else model(inputs)
            preds = torch.argmax(logits, dim=1).detach().cpu().numpy()
            hit += (preds == target_label).sum()
            total += len(preds)
    return hit / total if total > 0 else 0.0

def precompute_teacher_logits(model, dataloader, device):
    logger.info(f"Starting pre-computation of teacher logits on device {device}...")
    model.to(device)
    model.eval()
    
    cache = []
    with torch.no_grad():
        for batch in dataloader:
            inputs, labels = model.process(batch)
            gpu_inputs = {k: v.to(device) for k, v in inputs.items() if isinstance(v, torch.Tensor)}
            logits = model(gpu_inputs)['logits']
            cpu_inputs = {k: v.cpu() for k, v in inputs.items()}
            cpu_logits = logits.cpu()
            cpu_labels = labels.cpu()
            cache.append({'inputs': cpu_inputs, 'logits': cpu_logits, 'labels': cpu_labels})
            
    logger.info(f"Pre-computation finished. Cached {len(cache)} batches.")
    return cache

def main(config):
    seed_everything()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    project_name = "4090_LLM_Backdoor_Purification_Ablation"
    config_name = config['defender']['name'] + "_" + config['poison_dataset']['name'] + "_" + config["attacker"]["poisoner"]["name"] + "_" + config['victim']['model'] + "_" + str(config["unlearning"]["lambda_asr"])+ "_" + str(config["unlearning"]["lambda_acc"]) + "_" + str(config['unlearning']['lr'])

    logger.info(f'lambda_asr: {config["unlearning"]["lambda_asr"]}, lambda_acc: {config["unlearning"]["lambda_acc"]}, lr: {config["unlearning"]["lr"]}')

                
    wandb.init(
        project = project_name,
        name = config_name,
        config = config
    )

    UNLEARNING_EPOCHS = config['unlearning']['epochs'] 
    UNLEARNING_LR = config['unlearning']['lr'] 
    lambda_asr = config['unlearning']['lambda_asr']
    lambda_acc = config['unlearning']['lambda_acc']
    batch_size = config['defender']['batch_size']
    
    attack_type = config["attacker"]["poisoner"]['name']
    target_label = config['attacker']["poisoner"]["target_label"]
    attacker = load_attacker(config["attacker"])
    defender = load_defender(config["defender"])
    target_dataset = load_dataset(**config["target_dataset"])  

    dev_clean_data_path = './poison_data/{}/{}/{}/dev-clean.csv'.format(
        config['poison_dataset']['name'], target_label, attack_type)
    dev_poison_data_path = './poison_data/{}/{}/{}/dev-poison.csv'.format(
        config['poison_dataset']['name'], target_label, attack_type)
    dev_clean_data = load_poison_data(dev_clean_data_path)
    dev_poison_data = load_poison_data(dev_poison_data_path)
    dev_detect_data = {'dev-detect': dev_clean_data + dev_poison_data}

    logger.info("Loading backdoored model as the Teacher Model for pre-computation.")
    teacher_model = load_victim(config["victim"])
    loaded_params = torch.load(f"./models/dirty-{attack_type}-{config['attacker']['poisoner']['poison_rate']}-{config['poison_dataset']['name']}-{config['victim']['model']}/best.ckpt")
    if config['victim']['type'] == 'plm':
        teacher_model.load_state_dict(loaded_params)
    else:
        teacher_model.llm.load_state_dict(loaded_params)
    
    for param in teacher_model.parameters():
        param.requires_grad = False
    teacher_model.eval()


    cache_dir = './detection_cache'
    cache_filename = (
        f"{attack_type}_"
        f"{config['poison_dataset']['name']}_"
        f"{config['victim']['model']}.json"
    )
    cache_filepath = os.path.join(cache_dir, cache_filename)
    os.makedirs(cache_dir, exist_ok=True)

    if os.path.exists(cache_filepath):
        logger.info(f"Loading cached detection results from {cache_filepath}")
        with open(cache_filepath, 'r') as f:
            preds = json.load(f)
        
        if len(preds) != len(dev_detect_data['dev-detect']):
            logger.warning("Cached predictions length mismatch! Re-running detection.")
            _, preds = defender.eval_detect(model=teacher_model, clean_data=target_dataset, poison_data=dev_detect_data)
            logger.info(f"Saving new detection results to {cache_filepath}")
            with open(cache_filepath, 'w') as f:
                json.dump(preds.tolist() if isinstance(preds, np.ndarray) else preds, f)

    else:
        logger.info("No cached results found. Running poison detection...")
        _, preds = defender.eval_detect(model=teacher_model, clean_data=target_dataset, poison_data=dev_detect_data)
        
        logger.info(f"Saving new detection results to {cache_filepath}")
        with open(cache_filepath, 'w') as f:
            json.dump(preds.tolist() if isinstance(preds, np.ndarray) else preds, f)

    # _, preds = defender.eval_detect(model=teacher_model, clean_data=target_dataset, poison_data=dev_detect_data)
    # # preds = [1] * len(dev_detect_data['dev-detect'])
    poison_indices = [i for i, pred in enumerate(preds) if pred == 1]
    d_poison_list = [dev_detect_data['dev-detect'][i] for i in poison_indices]

    if not d_poison_list: 
        logger.warning("No poisoned samples detected. Skipping unlearning.")
        return

    poison_dataloader_for_cache = get_dataloader(d_poison_list, batch_size=batch_size)
    clean_dataloader_for_cache = get_dataloader(dev_clean_data, batch_size=batch_size)
    
    poison_cache = precompute_teacher_logits(teacher_model, poison_dataloader_for_cache, device)
    clean_cache = precompute_teacher_logits(teacher_model, clean_dataloader_for_cache, device)

    logger.info("Creating the Student Model as a copy of the Teacher.")
    student_model_base = load_victim(config["victim"])
    student_model_base.load_state_dict(teacher_model.state_dict())

    logger.info("Teacher logits cached. Unloading teacher model from VRAM.")
    del teacher_model
    torch.cuda.empty_cache() 

    pad_token_id = student_model_base.tokenizer.pad_token_id
    if pad_token_id is None: pad_token_id = student_model_base.tokenizer.eos_token_id
    original_forward = student_model_base.forward 
    student_model_base.forward = types.MethodType(universal_forward, student_model_base)

    if config['victim']['type'] == 'plm':
        student_model_base.config = student_model_base.plm.config
    else:
        student_model_base.config = student_model_base.llm.config

    logger.info("Injecting LoRA adapters into the Student Model.")
    if config['victim']['model'] == "bart-base":
        target_modules = ["q_proj", "k_proj", "v_proj", "out_proj"]
    elif config['victim']['model'] in ["Llama-3.2-3B-Instruct", "Qwen2.5-3B"]:
        target_modules = ["q_proj", "k_proj", "v_proj", "o_proj"]
    else:
        target_modules = ["query", "value", "key"]
    lora_config = LoraConfig(
        r=32,
        lora_alpha=64,
        target_modules=target_modules,
        lora_dropout=0.1,
        bias="none",
        task_type="SEQ_CLS" 
    )
    student_model = get_peft_model(student_model_base, lora_config)
    student_model.print_trainable_parameters()
    
    student_model.to(device)
    optimizer = torch.optim.AdamW(student_model.parameters(), lr=UNLEARNING_LR)
    num_training_steps = UNLEARNING_EPOCHS * len(poison_cache)
    num_warmup_steps = int(0.1 * num_training_steps)
    lr_scheduler = get_scheduler(
        name="linear", # "cosine"
        optimizer=optimizer,
        num_warmup_steps=num_warmup_steps,
        num_training_steps=num_training_steps
    )

    
    softmax_fn = Softmax(dim=1)
    log_softmax_fn = LogSoftmax(dim=1)
    kld_loss_fn = KLDivLoss(reduction='batchmean')
    criterion_ce = CrossEntropyLoss()
    
    val_poison_dataloader = get_dataloader(dev_poison_data, batch_size=batch_size)
    
    a = config['unlearning']['a'] 
    asr_threshold = config['unlearning']['asr_threshold']
    max_retries = config['unlearning']['max_retries']
    retry = 0
    best_asr = 1.0
    best_model_state = None

    logger.info(f"Found {len(d_poison_list)} poisoned samples. Starting unlearning process...")
    
    while retry < max_retries:
        student_model.train()
        
        clean_iterator = iter(clean_cache)
        
        for epoch in range(UNLEARNING_EPOCHS):
            for poison_data in poison_cache:
                try:
                    clean_data = next(clean_iterator)
                except StopIteration:
                    clean_iterator = iter(clean_cache)
                    clean_data = next(clean_iterator)
                
                optimizer.zero_grad()
                
                poison_inputs = poison_data['inputs']
                teacher_poison_logits = poison_data['logits']
                clean_inputs = clean_data['inputs']
                teacher_clean_logits = clean_data['logits']
                clean_labels = clean_data['labels']
                clean_labels = clean_labels.to(device)
                
                for key in poison_inputs:
                    if isinstance(poison_inputs[key], torch.Tensor):
                        poison_inputs[key] = poison_inputs[key].to(device)
                for key in clean_inputs:
                    if isinstance(clean_inputs[key], torch.Tensor):
                        clean_inputs[key] = clean_inputs[key].to(device)
                teacher_poison_logits = teacher_poison_logits.to(device)
                teacher_clean_logits = teacher_clean_logits.to(device)

                max_len = max(poison_inputs['input_ids'].size(1), clean_inputs['input_ids'].size(1))
                for key in poison_inputs:
                    if isinstance(poison_inputs[key], torch.Tensor) and poison_inputs[key].dim() == 2:
                        pad_value = 0 if 'mask' in key else pad_token_id
                        poison_inputs[key] = pad_to_max_length(poison_inputs[key], max_len, pad_value)
                        clean_inputs[key] = pad_to_max_length(clean_inputs[key], max_len, pad_value)

                student_poison_logits = student_model(**poison_inputs)['logits']
                student_clean_logits = student_model(**clean_inputs)['logits']

                loss_bkd = kld_loss_fn(
                    log_softmax_fn(student_poison_logits),
                    softmax_fn(teacher_poison_logits)
                )

                loss_cacc_protection = criterion_ce(student_clean_logits, clean_labels)
                
                combined_loss = -lambda_asr * loss_bkd + lambda_acc * loss_cacc_protection
                combined_loss.backward()
                torch.nn.utils.clip_grad_norm_(student_model.parameters(), max_norm=1.0)
                optimizer.step()
                lr_scheduler.step()
                
            logger.info(f"Unlearning Epoch {epoch+1}/{UNLEARNING_EPOCHS}, L_BKD: {loss_bkd.item():.4f}, loss_acc: {loss_cacc_protection.item():.4f}, lr: {optimizer.param_groups[0]['lr']}")

        logger.info("Training epoch finished. Cleaning cache before evaluation.")
        gc.collect()
        torch.cuda.empty_cache()
        
        asr = eval_asr(student_model, val_poison_dataloader, target_label, device)
        logger.info(f"[Validation] Poison ASR: {asr:.4f} (threshold: {asr_threshold})")
        
        if asr < best_asr:
            best_asr = asr
            best_model_state = {k: v.cpu() for k, v in student_model.state_dict().items()}
        
        if asr <= asr_threshold:
            logger.info(f"ASR has fallen below threshold, stopping training.")
            break
        else:
            for param_group in optimizer.param_groups:
                param_group['lr'] *= a
            logger.info(f"ASR does not meet the standard. Increase the learning rate to {optimizer.param_groups[0]['lr']:.6f} and retrain.")
            retry += 1
            if best_model_state is not None:
                student_model.load_state_dict(best_model_state)

    if best_model_state is not None:
        student_model.load_state_dict(best_model_state)

    student_model.eval()

    logger.info("Evaluating purified student model on test set.")
    purified_model = student_model.merge_and_unload()
    purified_model.forward = original_forward
    results = attacker.eval(purified_model, target_dataset)
    display_results(config, results)

    wandb.log({"results":results})
    wandb.finish()  


if __name__ == "__main__":
    for config_path in [
        './configs/badnets_config.json',
        './configs/addsent_config.json',
        './configs/style_config.json',
        './configs/syntactic_config.json'
    ]:
        with open(config_path, 'r') as f:
            config = json.load(f)

        for model, path in [
            ('bert-base-uncased', './victim_models/bert-base-uncased'),
            # ('bart-base', './victim_models/bart-base'),
            # ('Llama-3.2-3B-Instruct', "./victim_models/Llama-3.2-3B-Instruct"),
            # ('Qwen2.5-3B', "./victim_models/Qwen2.5-3B"),
        ]:
            
            config["victim"]['model'] = model
            config["victim"]['path'] = path
            if model in ('bert-base-uncased', 't5-base', 'bart-base'):
                config['victim']['type'] = 'plm'

            for dataset in [
                'sst-2', 
                # 'agnews', 
                # 'yelp'
            ]:      

                    lambda_acc = 1
                    lambda_asr = 0
    
                    config['poison_dataset']['name'] = dataset
                    config['target_dataset']['name'] = dataset
                    config["attacker"]['train']["poison_dataset"] = dataset
                    config['attacker']['train']['poison_model'] = model
    
                    config['attacker']['train']['epochs'] = 5
                    config['attacker']['poisoner']["poison_rate"] = 0.2
                    config['attacker']["poisoner"]["target_label"] = 1
                    config['attacker']['poisoner']['load'] = True
                    config['attacker']["train"]["batch_size"] = 4
                    config['victim']['num_classes'] = 2
                    if dataset == 'agnews':
                        config['victim']['num_classes'] = 4
                        config['attacker']["poisoner"]["target_label"] = 0
    
                    config.setdefault('defender', {})
                    config['defender']['name'] = 'ms'
                    config['defender']['data_name'] = dataset
                    config['defender']['attack_name'] = config['attacker']['poisoner']['name']
                    config['defender']['model_name'] = config["victim"]['model']
                    config['defender']['frr'] = 0.05
                    config['defender']['batch_size'] = 4
                    config['defender']["correction"] = False
                    config['defender']["pre"] = False
                    config['defender']["metrics"] = ["FRR", "FAR"]
                    if model in ('bert-base-uncased', 'bart-base'):
                        config['defender']['batch_size'] = 32
                
                    config['defender']['ss_layers'] = "all"
                    config['defender']['md_layers'] = "all"
                    config['defender']['md_top_k'] = 3
                    config['defender']['alpha'] = 0.9
                    config['defender']['visualize'] = False
                    config['defender']['purification'] = False
    
                    config.setdefault('unlearning', {})
                    config['unlearning']['epochs'] = 5
                    config['unlearning']['lr'] = 0.0005
                    config['unlearning']['lambda_asr'] = lambda_asr
                    config['unlearning']['lambda_acc'] = lambda_acc
                    config['unlearning']['a'] = 2
                    config['unlearning']['asr_threshold'] = 0.2 
                    config['unlearning']['max_retries'] = 1
    
                    torch.cuda.set_device(0)
                    config = set_config(config)
                    main(config)
    
                    logger.info("Cleaning up memory for the next run...")
                    gc.collect()
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()