training_config = {
    "temp": 1.0,
    "first_n_steps": 10,
    "c_puct": 1.25,
    "batch_size": 256,
    "dirichlet_alpha": 0.7,
}

env_config = {'row': 6, 
              'col': 7}

network_config = {"in_dim": 3,
                  "h_dim": 128,
                  "out_dim": env_config['col']}
