import json
import torch
import wandb
import os
from openbackdoor.attackers import load_attacker
from openbackdoor.data import load_dataset
from openbackdoor.victims import load_victim
from openbackdoor.utils import set_config, logger
from openbackdoor.utils.visualize import display_results
from accelerate import Accelerator
import gc
import multiprocessing

os.environ["WANDB_API_KEY"] = ''
os.environ["WANDB_MODE"] = "offline" 

def run_experiment(config):

    if config.get('use_accelerate', False): 
        accelerator = Accelerator()

    project_name = "4090_LLM_Backdoor_Attack"

    # dataset attack_type model_name poison_rate
    config_name = config['poison_dataset']['name'] + "_" + config["attacker"]["poisoner"]["name"] + "_" + config['victim']['model'] + "_" + str(config["attacker"]["poisoner"]["poison_rate"])

    wandb.init(
        project = project_name,
        name = config_name,
        config = config
    )    

    attacker = load_attacker(config["attacker"])
    victim = load_victim(config["victim"])
    target_dataset = load_dataset(**config["target_dataset"])
    poison_dataset = load_dataset(**config["poison_dataset"])
    logger.info("Train backdoored model on {}".format(config["poison_dataset"]["name"]))

    if config.get('use_accelerate', False): 
        attacker.poison_trainer.set_accelerator(accelerator)
    backdoored_model = attacker.attack(victim, poison_dataset)

    logger.info("Evaluate backdoored model on {}".format(config["target_dataset"]["name"]))

    results = attacker.eval(backdoored_model, target_dataset)
    display_results(config, results)

    wandb.log({"results":results})
    wandb.finish()

    logger.info(f"Experiment for {config['victim']['model']} finished. Cleaning up memory...")
    del victim, backdoored_model, attacker, target_dataset, poison_dataset, results
    if config.get('use_accelerate', False):
        del accelerator
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


if __name__ == "__main__":
    multiprocessing.set_start_method("spawn", force=True)

    for config_path in [
        # './configs/badnets_config.json',
        # './configs/addsent_config.json',
        # './configs/style_config.json',
        './configs/syntactic_config.json'
    ]:
        with open(config_path, 'r') as f:
            config = json.load(f)

        for model, path in [
            ('bert-base-uncased', './victim_models/bert-base-uncased'),
            ('bart-base', './victim_models/bart-base'),
            ('Llama-3.2-3B-Instruct', "./victim_models/Llama-3.2-3B-Instruct"),
            ('Qwen2.5-3B', "./victim_models/Qwen2.5-3B"),
        ]:
            
            config["victim"]['model'] = model
            config["victim"]['path'] = path
            config['attacker']['train']['visualize'] = False
            config['use_accelerate'] = False 
            if model in ('bert-base-uncased', 't5-base','bart-base'):
                config['victim']['type'] = 'plm'

            if model in ('Qwen2.5-3B-Instruct', 'Llama-3.2-3B-Instruct', 'Qwen2.5-3B'):
                config['use_accelerate'] = True 
                config['victim']['type'] = 'llm'
                config['attacker']['train']['name'] = 'lm'
                config['attacker']['train']['batch_size'] = 16
                config['attacker']['train']['gradient_accumulation_steps'] = 1

            for dataset in [
                # 'sst-2', 
                'agnews', 
                # 'yelp', 
                # 'hsol'
            ]:
                config['poison_dataset']['name'] = dataset
                config['target_dataset']['name'] = dataset
                config["attacker"]['train']["poison_dataset"] = dataset
                config['attacker']['train']['poison_model'] = model

                config['attacker']['train']['epochs'] = 5
                config['attacker']['poisoner']["poison_rate"] = 0.2
                config['attacker']["poisoner"]["target_label"] = 1
                config['attacker']['poisoner']['load'] = True
                config['victim']['num_classes'] = 2
                if dataset == 'agnews':
                    config['victim']['num_classes'] = 4
                    config['attacker']["poisoner"]["target_label"] = 0

                config = set_config(config)
                p = multiprocessing.Process(target=run_experiment, args=(config,))
                p.start()
                p.join() 