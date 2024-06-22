from torch.utils.data import Dataset
import pickle
import torch
import torch.nn as nn
import numpy as np
import os
import wandb
from omegaconf import OmegaConf
from pyDOE import lhs


    
class DataSampler:
    """
    A class for loading and sampling data for neural network training.

    Args:
        cfg (OmegaConf): The configuration for the dataset.

    Attributes:
        cfg (OmegaConf): The configuration for the dataset.
        idx (int): The current index for sampling batches.
        device (torch.device): The device to use for training.
        model_flag (str): The type of modelling being used.
        shuffle (bool): Whether to shuffle the data before splitting.
        split_ratio (float): The ratio of the training set over the validation and test sets.
        new_coll_points_flag (bool): Whether to use new collocation points.
        sampling (str): The sampling method for the initial conditions and possibly time.
        time (float): The time limit for the data.
        num_of_points (int): The number of points in the data.
        seed (int): The seed for the random number generator.
        data (list): The loaded data.
        input_dim (int): The dimension of the input data.
        total_trajectories (int): The total number of trajectories in the dataset.
        x (torch.Tensor): The input data.
        y (torch.Tensor): The target data.
        num_samples (int): The number of total samples.
        sample_per_traj (int): The number of samples per trajectory.
        x_train (torch.Tensor): The input data for the training set.
        x_val (torch.Tensor): The input data for the validation set.
        x_test (torch.Tensor): The input data for the testing set.
        y_train (torch.Tensor): The target data for the training set.
        y_val (torch.Tensor): The target data for the validation set (if used).
        y_test (torch.Tensor): The target data for the testing set (if used).
        x_train_col (torch.Tensor): The collocation points for the training set.
        minus_input (torch.Tensor): The value that was subtracted from the input data.(if transformed)
        divide_input (torch.Tensor): The value that was divided from the above result.(if transformed)
        minus_target (torch.Tensor): The value that was subtracted from the target data.(if transformed)
        divide_target (torch.Tensor): The value that was divided from the above result.(if transformed)

    Methods:
        load_data: Load data from file.
        next_batch: Get the next batch of data.
        data_input_target: Convert the loaded data into input and target data.
        data_input_target_limited: Convert the loaded data into input and target data with a time limit.
        train_test_split: Split the data into training and testing sets.
        train_val_test_split: Split the data into training, validation, and testing sets.
        find_norm_values: Find the min and max values of the data.
        find_std_values: Find the mean and std values of the data.
        define_minus_divide: Define the values to subtract and divide the data by.
        transform_data: Transform the data.
        detransform_data: De-transform the data.
        append_element_set: Append elements to a set of values.
        create_init_conditions_set: Create the initial conditions for the collocation points.
        create_col_points: Create the collocation points.
    """
    
    def __init__(self, cfg):
        self.cfg = cfg
        self.idx = 0
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model_flag = cfg.model.model_flag #  'SM_IB' or 'SM' or 'SM_AVR' or 'SM_GOV'
        self.shuffle = cfg.dataset.shuffle # Define whether to shuffle the data before splitting
        self.split_ratio = cfg.dataset.split_ratio # Define the ratio of the training set to the validation and test sets
        self.new_coll_points_flag = cfg.dataset.new_coll_points_flag # Define whether to use new collocation points
        if self.new_coll_points_flag:
            self.sampling = cfg.model.sampling # Define the sampling method for the initial conditions and possibly time
        self.time = cfg.time # Define the time limit for the data
        self.num_of_points = cfg.num_of_points # Define the number of points in the data
        self.seed = None if not hasattr(cfg.model, 'seed') else cfg.model.seed
        self.data, self.input_dim, self.total_trajectories = self.load_data() # Load the data from the file
        self.x, self.y  = self.data_input_target_limited(self.data, self.time) # Convert the loaded data into input and target data with a time limit
        #self.x, self.y = self.data_input_target(self.data)
        self.num_samples = self.x.shape[0] # number of total samples
        self.sample_per_traj = self.x.shape[0]/self.total_trajectories # number of samples per trajectory

        if self.cfg.dataset.validation_flag:
            self.x_train, self.x_val, self.x_test, self.y_train, self.y_val, self.y_test = self.train_val_test_split(self.x, self.y, self.split_ratio)
        else:
            self.x_train , self.x_test, self.y_train, self.y_test = self.train_test_split(self.x, self.y, self.split_ratio)
        if self.new_coll_points_flag:
            self.x_train_col = self.create_col_points().requires_grad_(True)
        else:
            self.x_train_col = self.x_train.clone().detach().requires_grad_(True)

        if self.cfg.dataset.transform_input != "None":
            self.minus_input, self.divide_input = self.define_minus_divide(self.x_train, self.x_train_col)
            self.minus_input = torch.nn.Parameter(self.minus_input, requires_grad=False)
            self.divide_input = torch.nn.Parameter(self.divide_input, requires_grad=False)
        if self.cfg.dataset.transform_output != "None":
            self.minus_target, self.divide_target = self.define_minus_divide(self.y_train, torch.empty(0))
            self.minus_target = torch.nn.Parameter(self.minus_target, requires_grad=False)
            self.divide_target = torch.nn.Parameter(self.divide_target, requires_grad=False)
         
        
    def load_data(self):
        """
        Load data from file.

        Returns:
            sol (list): The loaded data.
        """
        model_flag  = self.model_flag #  'SM_IB' or 'SM' or 'SM_AVR' or 'SM_GOV'
        number_of_dataset = self.cfg.dataset.number # Define the number of the dataset to load
        name = model_flag + '/dataset_v' + str(number_of_dataset) + '.pkl'
        dataset_path = os.path.join(self.cfg.dirs.dataset_dir, name) # Define the path to the dataset
        print("Loading data from: ", dataset_path)
        with open(dataset_path, 'rb') as f:
            sol = pickle.load(f)
        input_dim = len(sol[0])
        total_trajectories = len(sol)
        if self.shuffle:
            np.random.shuffle(sol) # shuffle the trajectories as they are ordered
        return sol, input_dim, total_trajectories

    def next_batch(self, batch_size):
        """
        Get the next batch of data.

        Args:
            batch_size (int): The size of the batch to sample.
        
        Returns:
            x_batch (torch.Tensor): The input data for the batch.
            y_batch (torch.Tensor): The target data for the batch.
        """
        perm = np.random.permutation(len(self.x_train))
        random_x_train = self.x_train[perm].clone()
        random_y_train = self.y_train[perm].clone()
        x_batch = random_x_train[:batch_size]
        y_batch = random_y_train[:batch_size]
        return x_batch, y_batch

    def data_input_target(self,data):
        """
        Convert the loaded data into input and target data.

        Args:
            data (list): The loaded data.
        
        Returns:
            x_train_list (torch.Tensor): The input data.
            y_train_list (torch.Tensor): The target data.
        """
        x_train_list = torch.tensor(())
        y_train_list = torch.tensor(())
        for training_sample in data:
            training_sample = torch.tensor(training_sample, dtype=torch.float32)
            y_train = training_sample[1:].T.clone().detach().requires_grad_(True)
            x_train = training_sample.T
            x_train[:,1:]=x_train[0][1:]
            #discrard the first row of x_train and y_train as they are the same
            #x_train = x_train[1:]
            #y_train = y_train[1:]
            x_train = x_train.clone().detach().requires_grad_(True)
            x_train_list = torch.cat((x_train_list, x_train), 0)
            y_train_list = torch.cat((y_train_list, y_train), 0)
        return x_train_list, y_train_list
    

    def data_input_target_limited(self, data, time_limit):
        """
        Convert the loaded data into input and target data and limit the time to time_limit.

        Args:
            data (list): The loaded data.
        
        Returns:
            x_train_list (torch.Tensor): The input data.
            y_train_list (torch.Tensor): The target data.
        """
        x_train_list = torch.tensor(())
        y_train_list = torch.tensor(())
        for training_sample in data:
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

        
    
    def train_test_split(self, x_train, y_train, split_ratio):
        """
        Split the data into training and testing sets.
        
        Args:
            x_train (torch.Tensor): The input data.
            y_train (torch.Tensor): The target data.
            split_ratio (float): The size of the training over test set.
            
        Returns:
            x_train (torch.Tensor): The input data for the training set.
            x_test (torch.Tensor): The input data for the testing set.
            y_train (torch.Tensor): The target data for the training set.
            y_test (torch.Tensor): The target data for the testing set.
            """

        split = int((self.total_trajectories) * split_ratio)*self.sample_per_traj
        x_train, x_test = x_train[:split], x_train[split:]
        y_train, y_test = y_train[:split], y_train[split:]
        return x_train, x_test, y_train, y_test
    
    def train_val_test_split(self, x_data, y_data, split_ratio):
        """
        Split the data into training, validation, and testing sets.
        
        Args:
            x_train (torch.Tensor): The input data.
            y_train (torch.Tensor): The target data.
            split_ratio (float): The ratio of the training set.

        Returns:
            x_train (torch.Tensor): The input data for the training set.
            x_val (torch.Tensor): The input data for the validation set.
            x_test (torch.Tensor): The input data for the testing set.
            y_train (torch.Tensor): The target data for the training set.
            y_val (torch.Tensor): The target data for the validation set.
            y_test (torch.Tensor): The target data for the testing set.
        """

        split = int(self.total_trajectories * split_ratio*self.sample_per_traj) # split trajectories and not just points into train, val and test
        val_split = int(self.total_trajectories * (10-split_ratio*10)/(2*10) *self.sample_per_traj) # multiply by 10 both denominator and numerator to get int values for sure
        x_train, x_val, x_test = x_data[:split], x_data[split:split+val_split], x_data[split+val_split:]
        y_train, y_val, y_test = y_data[:split], y_data[split:split+val_split], y_data[split+val_split:]
        print("Number of training samples: ", len(x_train), "Number of validation samples: ", len(x_val), "Number of testing samples: ", len(x_test))
        self.total_test_trajectories = int(self.total_trajectories * (10-split_ratio*10)/(2*10)) # number of test trajectories is given by the val_split and is equal to val trajectories
        return x_train, x_val, x_test, y_train, y_val, y_test
    
    def find_norm_values(self, data, data2):
        """
        This functions finds the min and the max value of the data
        """
        data_all = torch.cat((data.clone().detach(),data2.clone().detach()),0)
        min = data_all.min(dim=0).values
        max = data_all.max(dim=0).values
        if min.dim() == 0:
            min = min.unsqueeze(0)
        if max.dim() == 0:
            max = max.unsqueeze(0)
        return min, max
    
    def find_std_values(self, data,data2):
        """
        This functions finds the mean and the std value of the data
        """
        #find the mean and the std of the the total dataset = data+data2
        data_all = torch.cat((data.clone().detach(),data2.clone().detach()),0)
        mean = data_all.mean(dim=0)
        std = data_all.std(dim=0)
        noise = torch.rand_like(std) * 1e-7
        if mean.dim() == 0:
            mean = mean.unsqueeze(0)
        if std.dim() == 0:
            std = std.unsqueeze(0)
        std = std + noise
        std[std == 0] = 1 # Ensure std is not zero
        return mean, std
        
    def define_minus_divide(self, data, data_col):
        if self.cfg.dataset.transform_input == "Std" or self.cfg.dataset.transform_output == "Std":
            minus, divide = self.find_std_values(data, data_col)
        if self.cfg.dataset.transform_input == "MinMax" or self.cfg.dataset.transform_output == "MinMax" or self.cfg.dataset.transform_input == "MinMax2" or self.cfg.dataset.transform_output == "MinMax2":
            min, max = self.find_norm_values(data, data_col)
            minus = min
            divide = max - min
            noise = torch.rand_like(divide) * 1e-7
            divide = divide + noise
            divide[divide == 0] = 1
        return minus.to(self.device), divide.to(self.device)
    
    def transform_data(self, data, flag):
        """
        This function transform the data 
        """
        if flag == "Input":
            data = (data - self.minus_input) / self.divide_input
        elif flag == "Input2":
            mul = nn.Parameter(torch.tensor(2.0).to(self.device), requires_grad=False)
            min = nn.Parameter(torch.tensor(1.0).to(self.device), requires_grad=False)
            data = mul * (data - self.minus_input) / self.divide_input - min
        elif flag == "Output":
            data = (data - self.minus_target) / self.divide_target
        elif flag == "Output2":
            mul = nn.Parameter(torch.tensor(2.0).to(self.device), requires_grad=False)
            min = nn.Parameter(torch.tensor(1.0).to(self.device), requires_grad=False)
            data = mul * (data - self.minus_target) / self.divide_target - min
        else:
            raise Exception("Flag not implemented")
        return data

    def detransform_data(self, data, flag):
        """
        This function destandardize the data
        """  
        if flag == "Input":
            data = data * self.divide_input + self.minus_input
        elif flag == "Input2":
            mul = nn.Parameter(torch.tensor(2.0).to(self.device), requires_grad=False)
            min = nn.Parameter(torch.tensor(1.0).to(self.device), requires_grad=False)
            data = (data + min) * self.divide_input / mul + self.minus_input
        elif flag == "Output":
            data = data * self.divide_target + self.minus_target
        elif flag == "Output2":
            mul = nn.Parameter(torch.tensor(2.0).to(self.device), requires_grad=False)
            min = nn.Parameter(torch.tensor(1.0).to(self.device), requires_grad=False)
            data = (data + min) * self.divide_target / mul + self.minus_target
        else:
            raise Exception("Flag not implemented")
        return data
    
    
    def append_element_set(self, value_set, Value_range, num_ranges):
        seed = (self.seed if self.seed is not None else np.random.randint(0, 1000))
        if self.sampling=="Random":
            points = np.random.uniform(0, 1, num_ranges)
            points = points.reshape(-1, 1)
        elif self.sampling=="Linear":
            points = np.linspace(0, 1, num_ranges)
            points = points.reshape(-1, 1)
        elif self.sampling=="Lhs":
            points = lhs(n=1, samples=num_ranges)
        else:
            raise Exception("Sampling method not implemented")
        
        new_value_set = []
        iterations = len(value_set) if len(value_set)>1 else 1
        for j in range(iterations):
            if len(value_set)<1:
                values = (Value_range[0] + points * (Value_range[1] - Value_range[0]) if num_ranges > 1 else Value_range[0])
                new_value_set = [values][0].tolist()
            else:
                for i in points:
                    if isinstance(i, np.ndarray):
                        i = i.item()
                    value = (Value_range[0] + i * (Value_range[1] - Value_range[0]) if num_ranges > 1 else Value_range[0])
                    new_state = value_set[j].copy()
                    if isinstance(new_state, np.ndarray):
                        new_state = new_state.tolist()
                    new_state.extend([value])
                    new_value_set.append(new_state)
        return new_value_set

    def create_init_conditions_set(self):
        """
        Define the various initial conditions of the synchronous machine and return a matrix with all the possible combinations.

        Returns:
            list: A matrix with all the possible combinations of initial conditions.
        """
        #use the nn_init_cond.yaml file to create collocation points init conditions
        init_conditions_dir =  self.cfg.dirs.init_conditions_dir
        init_condition_bounds = self.cfg.model.init_condition_bounds
        init_conditions_path = os.path.join(init_conditions_dir, self.model_flag,"nn_init_cond"+str(init_condition_bounds)+".yaml")
        init_conditions = OmegaConf.load(init_conditions_path)
        # check if the initial conditions are in the correct format, e.g. if unique value then set iterations to 1
        for key in init_conditions.keys():
            if "iterations" not in key:
                if len(init_conditions[key])==1:
                    init_conditions[key+'_iterations']=1
                Theta_iterations = init_conditions.theta_iterations
        Omega_iterations = init_conditions.omega_iterations
        E_d_dash_iterations = init_conditions.E_d_dash_iterations
        E_q_dash_iterations = init_conditions.E_q_dash_iterations
        theta = init_conditions.theta
        omega = init_conditions.omega
        E_d_dash = init_conditions.E_d_dash
        E_q_dash = init_conditions.E_q_dash
        set_of_values = [theta, omega, E_d_dash, E_q_dash]
        iterations = [Theta_iterations, Omega_iterations, E_d_dash_iterations, E_q_dash_iterations]
        number_of_conditions = Theta_iterations * Omega_iterations * E_d_dash_iterations * E_q_dash_iterations
        if self.model_flag=="SM_AVR" or self.model_flag=="SM_AVR_GOV":
            R_F = init_conditions.R_F
            R_F_iterations = init_conditions.R_F_iterations
            V_r = init_conditions.V_r
            V_r_iterations = init_conditions.V_r_iterations
            E_fd = init_conditions.E_fd
            E_fd_iterations = init_conditions.E_fd_iterations
            set_of_values = set_of_values + [R_F, V_r, E_fd]
            iterations = iterations + [R_F_iterations, V_r_iterations, E_fd_iterations]
            number_of_conditions=number_of_conditions*R_F_iterations*V_r_iterations*E_fd_iterations
            if self.model_flag=="SM_AVR_GOV":
                P_m = init_conditions.P_m
                P_m_iterations = init_conditions.P_m_iterations
                P_sv = init_conditions.P_sv
                P_sv_iterations = init_conditions.P_sv_iterations
                set_of_values = set_of_values + [P_m, P_sv]
                iterations = iterations + [P_m_iterations, P_sv_iterations]
                number_of_conditions=number_of_conditions*P_m_iterations*P_sv_iterations
        print("Number of different initial conditions for collocation points: ", number_of_conditions)
        #wandb.log({"Number of different initial conditions for collocation points: ": number_of_conditions})
        print(set_of_values,"Set of values for init conditions")
        print(iterations,"Iterations per value")
        #wandb.log({"Set of values for init conditions: ": set_of_values})
        #wandb.log({"Iterations per value: ": iterations})
        init_condition_table = []   
        for k in range(len(set_of_values)):
            init_condition_table = self.append_element_set(init_condition_table, set_of_values[k], iterations[k])
        return init_condition_table

    def create_col_points(self):
        """
        Create the collocation points for the neural network training.

        Returns:
            torch.Tensor: The collocation points.
        """
        col_points = self.create_init_conditions_set() # create the initial conditions for the collocation points
        col_points = torch.tensor(col_points).float()

        time = torch.linspace(0, self.time, self.time*self.num_of_points).unsqueeze(1) # define time points 
        col_points_list = []

        for i in range(col_points.shape[0]):
            col_point = col_points[i].unsqueeze(1).repeat(1, time.shape[0]).T # repeat the collocation point for each time point
            col_point_time = torch.cat((time, col_point), dim=1) # add the time points to the collocation point
            col_points_list.append(col_point_time) # append the collocation points to the list

        col_points_tensor = torch.cat(col_points_list, dim=0)
        return col_points_tensor


class SM_Dataset(Dataset):
    """Define trajectory dataset class.
    Methods:
        __init__ : Initialize Trajectory class
        __getitem__ : Get item from dataset
        __len__ : Get length of dataset
    """
    
    def __init__(self, modelling, number_of_dataset):
        """Initialize Trajectory class.
        Args:
            data : Data to be used for training
        """
        self.modelling  =modelling #  'SM_IB' or 'SM' or 'SM_AVR' or 'SM_GOV'
        self.number_of_dataset = number_of_dataset 

        

    def __getitem__(self, idx):
        """Get item from dataset.
        Args:
            idx : Index of the dataset
        Returns:
            data : Data at index idx
        """
        return self.data[idx]
    
    def load_data(self):
        """Load data from file.
        Returns:
            data : Data from file
        """
        name = 'data/'+ self.modelling + '/dataset_v' + str(self.number_of_dataset) + '.pkl'
        with open(name, 'rb') as f:
            sol = pickle.load(f)

        return sol

    def __len__(self, data):
        """Get length of dataset.
        Returns:
            len(self.data) : Length of dataset
        """
        return len(data)

class TrajectorySampler:
    """
    A class for loading and sampling data for neural network training.

    Args:
        modelling (str): The type of modelling to use. Can be one of 'SM_IB', 'SM', 'SM_AVR', or 'SM_GOV'.
        number_of_dataset (int): The number of the dataset to load.

    Attributes:
        idx (int): The current index for sampling batches.
        modelling (str): The type of modelling being used.
        number_of_dataset (int): The number of the dataset being loaded.
        data (list): The loaded data.
        x (torch.Tensor): The input data.
        y (torch.Tensor): The target data.
        num_trajectories (int): The total number of trajectories in the dataset.

    Methods:
        load_data: Load data from file.
        next_batch: Get the next batch of data.
        data_input_target: Convert the loaded data into input and target data.
        data_input_target_test: Convert the loaded data into input and target data for testing.
        train_test_split: Split the data into training and testing sets.
        train_val_test_split: Split the data into training, validation, and testing sets.
    """
    
    def __init__(self, modelling, number_of_dataset):
        self.idx = 0
        self.modelling = modelling #  'SM_IB' or 'SM' or 'SM_AVR' or 'SM_GOV'
        self.number_of_dataset = number_of_dataset
        self.data , self.input_dim = self.load_data()
        self.num_trajectories = len(self.data)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.x, self.y = self.data_input_target(self.data)
        
    def load_data(self):
        """
        Load data from file.

        Returns:
            sol (list): The loaded data.
        """
        # replave path with cfg.dataset.path
        modelling  =self.modelling #  'SM_IB' or 'SM' or 'SM_AVR' or 'SM_GOV'
        number_of_dataset = self.number_of_dataset
        dataset_path = os.path.join(self.dataset_dir, modelling, '/dataset_v' + str(number_of_dataset) + '.pkl')
        print("Loading data from: ", dataset_path)
        with open(dataset_path, 'rb') as f:
            sol = pickle.load(f)
        input_dim = len(sol[0])
        return sol, input_dim

    def next_batch(self, batch_size):
        """
        Get the next batch of trajectories.

        Args:
            batch_size (int): The number of trajectories in the batch.
        
        Returns:
            x_batch (torch.Tensor): The input data for the batch.
            y_batch (torch.Tensor): The target data for the batch.
        """
        upper_limit = self.idx + batch_size
        down_limit = self.idx
        if upper_limit > self.num_trajectories:
            upper_limit2 = upper_limit - self.num_trajectories
            batch_data = self.data[:upper_limit2] + self.data[down_limit:]
            self.idx = upper_limit2
#            self.x, self.y = self.data_input_target(self.data)
        else:
            batch_data = self.data[down_limit:upper_limit]
            self.idx += batch_size
        return batch_data

    def data_input_target(self,data):
        """
        Convert the loaded data into input and target data.

        Args:
            data (list): The loaded data.
        
        Returns:
            x_train_list (torch.Tensor): The input data.
            y_train_list (torch.Tensor): The target data.
        """
        x_train_list = torch.tensor(())
        y_train_list = torch.tensor(())
        for training_sample in data:
            training_sample = torch.tensor(training_sample, dtype=torch.float32)
            y_train = training_sample[1:].T.clone().detach().requires_grad_(True)
            x_train = training_sample.T
            x_train[:,1:]=x_train[0][1:]
            #discrard the first row of x_train and y_train as they are the same
            #x_train = x_train[1:]
            #y_train = y_train[1:]
            x_train = x_train.clone().detach().requires_grad_(True)
            x_train_list = torch.cat((x_train_list, x_train), 0)
            y_train_list = torch.cat((y_train_list, y_train), 0)
        return x_train_list, y_train_list

    def data_input_target_limited(self, data, time_limit):
        """
        Convert the loaded data into input and target data and limit the time to time_limit.

        Args:
            data (list): The loaded data.
        
        Returns:
            x_train_list (torch.Tensor): The input data.
            y_train_list (torch.Tensor): The target data.
        """
        x_train_list = torch.tensor(())
        y_train_list = torch.tensor(())
        for training_sample in data:
            training_sample = torch.tensor(training_sample, dtype=torch.float32)
            y_train = training_sample[1:].T.clone().detach().requires_grad_(True)
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

        
    
    def train_test_split(self, x_train, y_train, test_size, shuffle):
        """
        Split the data into training and testing sets.
        
        Args:
            x_train (torch.Tensor): The input data.
            y_train (torch.Tensor): The target data.
            test_size (float): The size of the testing set.
            shuffle (bool): Whether to shuffle the data before splitting.
            
        Returns:
            x_train (torch.Tensor): The input data for the training set.
            x_test (torch.Tensor): The input data for the testing set.
            y_train (torch.Tensor): The target data for the training set.
            y_test (torch.Tensor): The target data for the testing set.
        """
        """
        if shuffle:
            perm = np.random.permutation(len(x_train))
            x_train = x_train[perm]
            y_train = y_train[perm]
        """
        split = int(len(x_train) * (1-test_size))
        x_train, x_test = x_train[:split], x_train[split:]
        y_train, y_test = y_train[:split], y_train[split:]
        return x_train, x_test, y_train, y_test
    
    def train_val_test_split(self, x_train, y_train, split_ratio, shuffle):
        """
        Split the data into training, validation, and testing sets.
        
        Args:
            x_train (torch.Tensor): The input data.
            y_train (torch.Tensor): The target data.
            split_ratio (float): The ratio of the training set.
            shuffle (bool): Whether to shuffle the data before splitting.

        Returns:
            x_train (torch.Tensor): The input data for the training set.
            x_val (torch.Tensor): The input data for the validation set.
            x_test (torch.Tensor): The input data for the testing set.
            y_train (torch.Tensor): The target data for the training set.
            y_val (torch.Tensor): The target data for the validation set.
            y_test (torch.Tensor): The target data for the testing set.
        """
        if shuffle:
            perm = np.random.permutation(len(x_train))
            x_train = x_train[perm]
            y_train = y_train[perm]
        split = int(len(x_train) * split_ratio)
        val_split = int(len(x_train) * (1-split_ratio)/2) 
        x_train, x_val, x_test = x_train[:split], x_train[split:split+val_split], x_train[split+val_split:]
        y_train, y_val, y_test = y_train[:split], y_train[split:split+val_split], y_train[split+val_split:]
        return x_train, x_val, x_test, y_train, y_val, y_test
    
    def normalize_data(self, data0):
        """
        This function standardize the data
        """  
        data = data0.clone()
        data = (data - data.mean()) / data.std()
    
        return data, data.mean(), data.std()
    
    def denormalize_data(self, data0, mean, std):
        """
        This function destandardize the data
        """  
        data = data0.clone()
        data = data * std + mean
    
        return data
    
    def adjust_target(self, input, target):
        """
        This function adjust the target data to be the x[i]+ x[0]*target[i] instead of target[i]
        """  
        
        return input[:,1:] + input[:,0].view(-1,1)*target
    
    def adjust_target_2(self, input, target):
        """
        This function adjust the target data to be the x[i]+ x[0]*target[i] instead of target[i]
        """
        return input[:,1:] + target
    
    def tanh_input(self, input):
        """
        This function adjust the input data to be the tanh(x) instead of x
        """  
        
        output = torch.tanh(input)
        output = output.clone().detach()
        return output
    
    def adjust_target_inv(self, input, target):
        """
        This function adjust the target data to be the x[i]+ x[0]*target[i] instead of target[i]
        """  
        
        output = (target- input[:,1:])/input[:,0].view(-1,1) 
        output = output.clone().detach()
        return output
    
    def adjust_target_inv_2(self, input, target):
        """
        This function adjust the target data to be the x[i]+ x[0]*target[i] instead of target[i]
        """  
        
        output = target - input[:,1:]
        output = output.clone().detach()
        return output
    
    def play_with_data(self, data, time_limit, normalize_flag, tanh_flag, adjust_flag):
        x_data_list, y_data_list = self.data_input_target_limited(data, time_limit)
        if normalize_flag:
            x_data_list_n, mean, std = self.normalize_data(x_data_list) 
        else:
            x_data_list_n = x_data_list
        if tanh_flag:
            x_data_list_n [:,0] = self.tanh_input(x_data_list_n[:,0])
        if adjust_flag:
            y_data_list_new = self.adjust_target(x_data_list,y_data_list).to(self.device)
        else:
            y_data_list_new = y_data_list
        
        x_data_list_new = x_data_list_n.clone().detach().to(self.device).requires_grad_(True)
        y_data_list_new = y_data_list_new.clone().detach().to(self.device)
        if normalize_flag:
            return x_data_list_new, y_data_list_new, mean, std
        return x_data_list_new, y_data_list_new