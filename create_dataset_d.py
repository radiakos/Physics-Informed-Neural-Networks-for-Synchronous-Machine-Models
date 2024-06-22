from src.functions import *
from src.params import *
import torch
import wandb
import hydra
from src.dataset.create_dataset_functions import SM_modelling
import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"



# Load config file using hydra
@hydra.main(config_path="../Pinn-Thesis/src/conf", config_name="setup_dataset.yaml",version_base=None)
def main(config):
    # Initialize wandb
    run = wandb.init(project=config.wandb.project)
    log_data_metrics_to_wandb(run, config)
    SM_model=SM_modelling(config) # Create an instance of the class CreateDataset
    machine_params=SM_model.define_machine_params() # Define the parameters of the synchronous machine
    system_params=SM_model.define_system_params() # Define the parameters of the power system
    init_conditions=SM_model.create_init_conditions_set3() # Define the initial conditions of the system
    print("Is cuda available?", torch.cuda.is_available())
    solution = SM_model.solve_sm_model(machine_params, system_params, init_conditions) # Solve the model for the various initial conditions
    # Save the dataset
    SM_model.save_dataset(solution) # Save the dataset
    return None

if __name__ == "__main__":
    main()

