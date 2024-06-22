import os
from src.nn.nn_actions import NeuralNetworkActions
from src.functions import *
from omegaconf import OmegaConf
import wandb

def train(config=None):
    # Load configuration from YAML file
    run = wandb.init(config=config)
    config = run.config
    cfg = OmegaConf.load("src/conf/setup_dataset_nn.yaml")
    cfg.nn.type = config.nn_type
    if cfg.nn.type == "DynamicNN":
        cfg.nn.num_epochs = 10000
    if config.optimizer == "LBFGS":
        lbfgs_iter = 20 if cfg.nn.type == "DynamicNN" else 20
        cfg.nn.num_epochs = int(cfg.nn.num_epochs/lbfgs_iter)
    cfg.nn.optimizer = config.optimizer
    cfg.nn.update_weight_method = config.update_weight_method

    cfg.dataset.transform_input = config.transform_input
    cfg.dataset.transform_output = config.transform_output
    network = NeuralNetworkActions(cfg)

    
    log_data_metrics_to_wandb(run, cfg)
    log_pinn_metrics_to_wandb(run, cfg)

    num_of_data = config.num_of_data
    #num_of_skip_data_points = 23
    #num_of_skip_col_points = 5
    num_of_skip_data_points = config.num_of_skip_data_points
    num_of_skip_col_points = config.num_of_skip_col_points
    num_of_skip_val_points = 4
    weight_data = config.weight_data
    weight_dt = config.weight_dt
    weight_pinn = config.weight_pinn
    
    # Adjust the maximum number of iterations here

    network.pinn_train2(weight_data, weight_dt, weight_pinn, num_of_data, num_of_skip_data_points, num_of_skip_col_points, num_of_skip_val_points, run)
        

    wandb.finish()

if __name__ == "__main__":
    # Load configuration from YAML file
    cfg = OmegaConf.load("src/conf/setup_dataset_nn.yaml")

    # Sweep configuration
    sweep_config = {
        "method": "random", #choose between grid and random
        "metric": {"name": "Val_loss", "goal": "minimize"},
        "parameters": {
            "weight_data": {"values": [0,1]},
            "weight_dt": {"values": [0,1e-4]},
            "weight_pinn": {"values": [0,1e-5]},
            "nn_type" :{"values": ["DynamicNN"]}, #,"PinnAA"
            "optimizer" : {"values": ["LBFGS"]},
            "update_weight_method" : {"values": ["Static","Dynamic","Sam"]},#,"Dynamic"
            "num_of_skip_data_points" :{"values": [23]},
            "num_of_skip_col_points" :{"values": [5]},#5,9, 13,19,23
            "num_of_data" :{"values": [4000000, 6000000]},
            "transform_input":{"values": ["None","Std","MinMax","MinMax2"]},
            "transform_output":{"values": ["None","Std","MinMax","MinMax2"]},
            # Add other parameters from setup_dataset_nn.yaml as needed
        }
        
    }

    # Initialize sweep
    sweep_id = wandb.sweep(sweep_config, project="your_project_name")

    # Run the sweep
    wandb.agent(sweep_id, function=train,count=50)
    wandb.finish()