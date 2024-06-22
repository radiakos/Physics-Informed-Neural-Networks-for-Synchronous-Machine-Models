import torch
import torch.nn as nn

from omegaconf import OmegaConf
import wandb
from src.nn.nn_model import Net, Network, PinnA
import os

def find_files(cfg, dir):
    files = []
    for folder in dir:
        directory = os.path.join(cfg.dirs.model_dir,folder)
        for file in os.listdir(directory):
            if file.endswith(".pth"):
                file = os.path.join(folder, file)
                files.append(file)
    return files

def define_model_from_name(name):
    name_list = ["DynamicNN", "PinnA", "PinnAA", "PinnB"]
    #CHECK if name_list is in the name
    for n in name_list:
        if n in name:
            model_name = n
            return model_name   

def check_transform(name):
    transform_input_list = ["MinMax", "Std", "MinMax2"]
    for t1 in transform_input_list:
        if t1 in name:
            return True
        else:
            return False 
        
def check_sm_modelling(name, cfg):
    sm_modelling_list = ["SM_AVR_GOV", "SM_AVR", "SMIB", "SM"]
    for t1 in sm_modelling_list:
        if t1 in name:
            if t1 == cfg.model.model_flag:
                print(t1)
                return False
            else:
                return True
        else:
            return False

def define_nn_model(cfg, input_dim, output_dim):
    """
    This function defines the neural network model
    """
    if cfg.nn.type == "StaticNN":
        model = Net(input_dim, cfg.nn.hidden_dim, output_dim)
    elif cfg.nn.type == "DynamicNN" or cfg.nn.type == "PinnB" or cfg.nn.type == "PinnA":
        model = Network(input_dim, cfg.nn.hidden_dim, output_dim, cfg.nn.hidden_layers)
    elif cfg.nn.type == "PinnAA":
        model = PinnA(input_dim, cfg.nn.hidden_dim, output_dim, cfg.nn.hidden_layers)
    else:
        raise ValueError("Invalid nn type specified in the configuration.")
    return model

def forward_pass(model, data_network, input):
    """
    This function calculates the output of the neural network model, input is given as time and the other input columns
    """
    time = input[:,0].unsqueeze(1) # get the time column
    no_time = input[:,1:]
    model.eval()
    y_pred = model.forward(input)
    if data_network.cfg.nn.type == "PinnA":
        if data_network.cfg.dataset.transform_input == "None":
            return no_time + y_pred*time
        minus = data_network.data_loader.minus_input.clone().detach().to(data_network.device)
        divide = data_network.data_loader.divide_input.clone().detach().to(data_network.device)
        if data_network.cfg.dataset.transform_input == "MinMax2":
            div = nn.Parameter(torch.tensor(2.0), requires_grad=False)
            plus = nn.Parameter(torch.tensor(1.0), requires_grad=False)
            return ((no_time + plus) * divide[1:] / div + minus[1:]) + y_pred*((time + plus) * divide[0] / div + minus[0])
        return (no_time* divide[1:] + minus[1:]) + y_pred*(time* divide[0] + minus[0])
    if data_network.cfg.nn.type == "PinnB":
        return no_time + y_pred
    if data_network.cfg.nn.type == "DynamicNN" or data_network.cfg.nn.type == "PinnAA":
        return y_pred
    else:
        raise Exception('Enter valid NN type! (zeroth_order or first_order')
        
def load_model(model, cfg, name=None):
    """
    Load neural network model weights from the model_dir.

    Args:
        name (str): name of the model
    """
    # Load model from the model_dir
    model_dir = cfg.dirs.model_dir
    if not os.path.exists(model_dir) or len(os.listdir(model_dir)) == 0:
        raise Exception("No model found in the model_dir, please correct path and name of the model from the bucket first")
    if name is None:
        # Find first model in the model_dir
        name = os.listdir(model_dir)[0]
        if name == '.gitkeep':
            if len(os.listdir(model_dir)) == 1:
                raise Exception("No model found in the model_dir, please correct path and name of the model from the bucket first")
            name = os.listdir(model_dir)[1]
        print("Load model:", name)
    model_path = os.path.join(model_dir, name)
    if not os.path.exists(model_path):
        raise Exception("No model found in the model_dir, please correct path and name of the model from the bucket first")
    #check if torch load
    model_data = torch.load(model_path)
    model.load_state_dict(model_data['model_state_dict'])
    return None

    
    

def data_input_target_limited(solution, time_limit):
    """
    Convert the loaded data into input and target data and limit the time to time_limit.

    Args:
        data (list): The loaded data.
    
    Returns:
        x_train_list (torch.Tensor): The input data.
        y_train_list (torch.Tensor): The target data.
    """
    dataset = []
    for i in range(len(solution)):
        r = [solution[i].t]  # append time to directory
        for j in range(len(solution[i].y)):
            r.append(solution[i].y[j])  # append the solution at each time step
        dataset.append(r)
    x_train_list = torch.tensor(())
    y_train_list = torch.tensor(())
    for training_sample in dataset:
        training_sample = torch.tensor(training_sample, dtype=torch.float32) # convert the trajectory to tensor
        y_train = training_sample[1:].T.clone().detach().requires_grad_(True) # target data
        training_sample_l = training_sample.T
        training_sample_l = training_sample_l[training_sample_l[:,0]<=time_limit].T # limit the time to time_limit
        y_train = training_sample_l[1:].T.clone().detach()
        x_train = training_sample_l.T
        x_train[:,1:]=x_train[0][1:]
        #discrard the first row of x_train and y_train as they are the same
        #x_train = x_train[1:]
        #y_train = y_train[1:]
        x_train = x_train.clone().detach().requires_grad_(True)
        x_train_list = torch.cat((x_train_list, x_train), 0)
        y_train_list = torch.cat((y_train_list, y_train), 0)
    return x_train_list, y_train_list

def forward_pass_b(model,x_train):
    """
    Perform a forward pass of the model.

    Args:
        model (torch.nn.Module): The neural network model.
        x_train (torch.Tensor): The input data.
    
    Returns:
        y_pred (torch.Tensor): The predicted target data.
    """
    y_pred = model(x_train)
    no_time = x_train[:,1:]
    return no_time+y_pred

def forward_pass_a(model,x_train):
    """
    Perform a forward pass of the model.

    Args:
        model (torch.nn.Module): The neural network model.
        x_train (torch.Tensor): The input data.
    
    Returns:
        y_pred (torch.Tensor): The predicted target data.
    """
    y_pred = model(x_train)
    time = x_train[:,0].unsqueeze(1) # get the time column
    no_time = x_train[:,1:]
    return no_time+time*y_pred


def predict(name, type, x_train_list,cfg):
    cfg.nn.type = type
    input_dim = x_train_list.shape[1]
    output_dim = x_train_list.shape[1]-1
    model = define_nn_model(cfg, input_dim, output_dim)
    load_model(model, cfg, name)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    if type == "PinnA":
        y_pred = forward_pass_a(model,x_train_list)
    elif type == "PinnB":
        y_pred = forward_pass_b(model,x_train_list)
    else:
        y_pred = model.forward(x_train_list)
    return y_pred.detach().cpu().numpy()
