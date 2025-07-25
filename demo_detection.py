import json
import os
import wandb
from openbackdoor.defenders import load_defender
import torch
from openbackdoor.attackers import load_attacker
from openbackdoor.data import load_dataset
from openbackdoor.victims import load_victim
from openbackdoor.utils import set_config, logger
from openbackdoor.utils.visualize import display_results

os.environ["WANDB_API_KEY"] = ''
os.environ["WANDB_MODE"] = "offline" 

def main(config):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    project_name = "4090_Detection_Ablation"

    config_name = config['defender']['name'] + "_" + config['poison_dataset']['name'] + "_" + config["attacker"]["poisoner"]["name"] + "_" + config['victim']['model'] + "_" + str(config["defender"]["alpha"])

    wandb.init(
        project = project_name,
        name = config_name,
        config = config
    )
    
    attack_type = config["attacker"]["poisoner"]['name']
    attacker = load_attacker(config["attacker"])
    victim = load_victim(config["victim"])
    target_dataset = load_dataset(**config["target_dataset"])

    defender = load_defender(config["defender"])

    target_dataset = load_dataset(**config["target_dataset"])
    logger.info("Load backdoored model on {}".format(config["poison_dataset"]["name"]))
    loaded_backdoored_model_params = torch.load('./models/dirty-{}-{}-{}-{}/best.ckpt'.format(
        attack_type,
        config['attacker']["poisoner"]["poison_rate"],
        config['poison_dataset']['name'],
        config["victim"]['model']
        ),
        map_location=device
    )
    if config['victim']['type'] == 'plm':
        victim.load_state_dict(loaded_backdoored_model_params)
    else:
        victim.llm.load_state_dict(loaded_backdoored_model_params)
    victim.eval()
    backdoored_model = victim
    backdoored_model.to(device)
    logger.info("Evaluate {} on the backdoored model".format(config['defender']['name']))
    results = attacker.eval(backdoored_model, target_dataset, defender)
    display_results(config, results)

    wandb.log({"results":results})
    wandb.finish()

    

if __name__ == "__main__":
    for config_path in [
        # './configs/badnets_config.json',
        # './configs/addsent_config.json',
        './configs/style_config.json',
        './configs/syntactic_config.json'
    ]:
        with open(config_path, 'r') as f:
            config = json.load(f)

        for model, path in [
            # ('bert-base-uncased', './victim_models/bert-base-uncased'),
            # ('bart-base', './victim_models/bart-base'),
            ('Llama-3.2-3B-Instruct', "./victim_models/Llama-3.2-3B-Instruct"),
            # ('Qwen2.5-3B', "./victim_models/Qwen2.5-3B"),
            
        ]:
            
            config["victim"]['model'] = model
            config["victim"]['path'] = path
            if model in ('bert-base-uncased', 't5-base', 'bart-base'):
                config['victim']['type'] = 'plm'
            else:
                config['victim']['type'] = 'llm'

            for dataset in [
                'sst-2', 
                # 'agnews', 
                # 'yelp'
            ]:
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

                for defender_name in [
                    # 'badacts', 
                    # 'strip', 
                    # 'dan', 
                    'ms'
                ]:
                    config.setdefault('defender', {})
                    config['defender']['name'] = defender_name
                    config['defender']['data_name'] = dataset
                    config['defender']['attack_name'] = config['attacker']['poisoner']['name']
                    config['defender']['model_name'] = config["victim"]['model']
                    config['defender']['frr'] = 0.05
                    config['defender']['batch_size'] = 4
                    config['defender']["correction"] = False
                    config['defender']["pre"] = False
                    config['defender']["metrics"] = ["FRR", "FAR"]
                    if defender_name == 'ms':  
                        config['defender']['ss_layers'] = "all"
                        config['defender']['md_layers'] = "all"
                        config['defender']['md_top_k'] = 3
                        config['defender']['alpha'] = 1
                        config['defender']['visualize'] = False
                    elif defender_name == 'badacts':
                        config['defender']["delta"] = 3
                    elif defender_name == 'onion':
                        config['defender']['threshold'] = 0
                    elif defender_name == 'strip':
                        config['defender']['repeat'] = 5
                        config['defender']['swap_ratio'] = 0.5
                        config['defender']['use_oppsite_set'] = False

                    torch.cuda.set_device(0)
                    config = set_config(config)
                    main(config)