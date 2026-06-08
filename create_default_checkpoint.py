import torch
import os
def create_default_checkpoint():
    from models.dhpen_network import DHPENNetwork
    config = {
        'input_dim': 59,
        'prototype_dim': 128,
        'n_way': 5,
        'k_shot': 5,
        'n_query': 15,
        'inner_lr': 0.01,
        'meta_lr': 1e-4,
        'adaptation_steps': 5,
        'alpha_fusion': 0.7,
        'beta_update': 0.9,
        'temperature': 18.0,
        'device': 'cpu',
        'feature_dim': 59,
        'use_complexity_estimator': True,
        'use_hierarchy_constructor': True,
        'use_evolution_engine': True,
        'use_global_memory': True,
        'checkpoint_dir': './checkpoints',
        'modal_indices': {
            'basic': list(range(10)),
            'traffic': list(range(10, 25)),
            'connection': list(range(25, 40)),
            'content': list(range(40, 59))
        },
        'modal_dims': {'basic': 10, 'traffic': 15, 'connection': 15, 'content': 19},
        'modal_types': ['basic', 'traffic', 'connection', 'content']
    }
    model = DHPENNetwork(config)
    os.makedirs('./checkpoints', exist_ok=True)
    torch.save(model.state_dict(), './checkpoints/model_checkpoint.pth')
    print(": ./checkpoints/model_checkpoint.pth")
    torch.save(model.state_dict(), './checkpoints/model_best.pth')
    print(": ./checkpoints/model_best.pth")
if __name__ == '__main__':
    create_default_checkpoint()
