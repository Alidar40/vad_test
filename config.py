import os
import yaml

from dotenv import load_dotenv

load_dotenv()


config = yaml.safe_load(open("config.yaml", "rb"))
config['frame_size'] = int(config['sample_rate'] * (config['frame_size_ms'] / 1000.0))

if config['wandb']['name'] == 'as_model':
    config['wandb']['name'] = config['model']

config["cobra_access_key"] = os.environ['COBRA_ACCESS_KEY']

