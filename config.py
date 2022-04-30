import yaml


config = yaml.safe_load(open("config.yaml", "rb"))
config['frame_size'] = int(config['sample_rate'] * (config['frame_size_ms'] / 1000.0))

if config['wandb']['name'] == 'as_model':
    config['wandb']['name'] = config['model']
