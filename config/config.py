import yaml
import os
def load_config(config_path):
    if not os.path.exists(config_path):
        raise FileNotFoundError(f": {config_path}")
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    if 'training' in config:
        if 'learning_rate' in config['training']:
            config['training']['learning_rate'] = float(config['training']['learning_rate'])
        if 'weight_decay' in config['training']:
            config['training']['weight_decay'] = float(config['training']['weight_decay'])
    if 'drift' in config and 'adaptation' in config['drift']:
        if 'smoothing' in config['drift']['adaptation']:
            config['drift']['adaptation']['smoothing'] = float(config['drift']['adaptation']['smoothing'])
        if 'retention_weight' in config['drift']['adaptation']:
            config['drift']['adaptation']['retention_weight'] = float(config['drift']['adaptation']['retention_weight'])
    if 'drift' in config:
        if 'bandwidth' in config['drift']:
            config['drift']['bandwidth'] = float(config['drift']['bandwidth'])
        if 'threshold' in config['drift']:
            config['drift']['threshold'] = float(config['drift']['threshold'])
    if 'llm' in config:
        if 'distillation_temperature' in config['llm']:
            config['llm']['distillation_temperature'] = float(config['llm']['distillation_temperature'])
    return config
