import torch
import torch.nn as nn
import torch.optim as optim
import os
import matplotlib.pyplot as plt
from src.nn.nn_dataset import DataSampler
from src.nn.nn_model import Net, Network, PinnA, FullyConnectedResNet, Kalm
from src.dataset.create_dataset_functions import SM_modelling
from src.functions import *
from src.nn.early_stopping import EarlyStopping
from src.nn.relobralo import ReLoBRaLoLoss
import wandb

class NeuralNetworkActions():
    """
    A class used to define the actions of the neural network model

    Attributes
    ----------
    cfg (dict) : configuration file
    input_dim (int) : number of input features
    hidden_dim (int) : number of hidden neural network layers
    output_dim (int) : number of output features
    learning_rate (float) : learning rate of the optimizer
    model (Net) : neural network model class
    criterion (nn.Module) : loss function
    optimizer (optim) : optimizer
    scheduler (optim) : learning rate scheduler
    SM_model (SM_modelling) : class for creating the synchronous machine model
    machine_params (dict) : parameters of the synchronous machine
    system_params (dict) : parameters of the power system
    solver (CreateSolver) : class for solving the synchronous machine model
    flag_for_modelling (bool) : flag for using the synchronous machine model
    device (torch.device) : device to run the model
    
    Methods
    -------
    define_nn_model()
        This function defines the neural network model
    custom_loss(loss_name)
        This function defines the loss function
    custom_optimizer(optimizer_name, learning_rate)
        This function defines the optimizer
    custom_learning_rate(lr_name)
        This function defines the learning rate scheduler
    custom_weight_loss_updated(weight_data, weight_dt, weight_pinn, data_loss, dt_loss, pinn_loss, epoch)
        This function updates the weights of the loss functions
    weight_init(module, init_name)
        This function initializes the weights of the neural network model
    test(x_test)
        This function tests the neural network model
    plot(x_train, y_train, var=0)
        This function plots the data for a specific variable
    plot_all(x_train, y_train)
        This function plots all the data in pairs
    plot_all_dt(x_train, y_train)
        This function plots the derivative of all the data in pairs
    forward_nn(time, no_time)
        This function calculates the output of the neural network model, input is given as time and the other input columns
    forward_pass(x_train)
        This function calculates the output of the neural network model, input is given as the one whole tensor


     
    """
    def __init__(self, cfg):
        self.cfg = cfg
        set_random_seeds(cfg.seed) # set all seeds
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')  

        self.data_loader = DataSampler(cfg)
        self.input_dim = self.data_loader.input_dim # The input dimension is the number of input features
        self.output_dim = self.input_dim-1 # The output dimension is the input dimension minus the time column

        self.model = self.define_nn_model() # Create an instance of the class Net
        self.weight_init(self.model, cfg.nn.weight_init) # Initialize the weights of the Net
        self.criterion = self.custom_loss(cfg.nn.loss_criterion) # Define the loss function
        self.criterion_mae = nn.L1Loss() # Define the MAE loss for testing
        self.optimizer = self.custom_optimizer(cfg.nn.optimizer, cfg.nn.lr) # Define the optimizer
        self.scheduler = self.custom_learning_rate(cfg.nn.lr_scheduler) # Define the learning rate scheduler
        
        self.SM_model= SM_modelling(cfg) # Create an instance of the class CreateDataset
        self.machine_params=self.SM_model.define_machine_params() # Define the parameters of the synchronous machine
        self.system_params=self.SM_model.define_system_params() # Define the parameters of the power system
        self.solver = self.SM_model.create_solver(self.machine_params, self.system_params) # Create an torch(based on cfg.model.torch) instance of the class CreateSolver
        
        self.model = self.model.to(self.device)
        self.early_stopping = EarlyStopping(patience=cfg.nn.early_stopping_patience, verbose=True, delta=cfg.nn.early_stopping_min_delta)
        if cfg.nn.update_weight_method=="ReLoBRaLo":
            self.relobralo_loss = ReLoBRaLoLoss()


    def setup_nn(self):
        self.model = self.define_nn_model() # Create an instance of the class Net
        self.weight_init(self.model, self.cfg.nn.weight_init) # Initialize the weights of the Net
        self.criterion = self.custom_loss(self.cfg.nn.loss_criterion) # Define the loss function
        if self.cfg.nn.optimizer == "Hybrid":  # Define the optimizer
            self.optimizer = self.custom_optimizer("Adam", self.cfg.nn.lr)
            self.optimizer2 = self.custom_optimizer("LBFGS", self.cfg.nn.lr)
        else:
            self.optimizer = self.custom_optimizer(self.cfg.nn.optimizer, self.cfg.nn.lr)
        self.scheduler = self.custom_learning_rate(self.cfg.nn.lr_scheduler) # Define the learning rate scheduler
        self.model = self.model.to(self.device)
        self.early_stopping = EarlyStopping(patience=self.cfg.nn.early_stopping_patience, verbose=True, delta=self.cfg.nn.early_stopping_min_delta)
        if self.cfg.nn.update_weight_method=="ReLoBRaLo":
            self.relobralo_loss = ReLoBRaLoLoss()
        return


    def define_nn_model(self):
        """
        This function defines the neural network model
        """
        print("########HERE#####",self.cfg.nn.type)
        if self.cfg.nn.type == "KAN": # Static architecture of the neural network
            print(self.input_dim, self.cfg.nn.hidden_dim, self.output_dim)
            model = Kalm(self.input_dim, self.cfg.nn.hidden_dim, self.output_dim)
            # model.speed()
        if self.cfg.nn.type == "StaticNN": # Static architecture of the neural network
            model = Net(self.input_dim, self.cfg.nn.hidden_dim, self.output_dim)
        if self.cfg.nn.type == "DynamicNN" or self.cfg.nn.type == "PinnB" or self.cfg.nn.type == "PinnA": # Dynamic architecture of the neural network
            model = Network(self.input_dim, self.cfg.nn.hidden_dim, self.output_dim, self.cfg.nn.hidden_layers)
        if self.cfg.nn.type == "PinnAA": # Dynamic architecture of the neural network with the PinnA architecture for the output
            model = PinnA(self.input_dim, self.cfg.nn.hidden_dim, self.output_dim, self.cfg.nn.hidden_layers)
        #############KAN###############
        if self.cfg.nn.type == "ResNet":
            num_blocks=2
            num_layers_per_block=2
            model = FullyConnectedResNet(self.input_dim, self.cfg.nn.hidden_dim, self.output_dim, num_blocks, num_layers_per_block)
        return model

    def custom_loss(self, loss_name):
        """
        This function defines the loss function
        
        Args:
            loss_name (str): name of the loss function
        
        Returns:
            criterion (nn.Module): loss function
        """
        if loss_name == 'MSELoss': # Mean Squared Error Loss
            criterion = nn.MSELoss()
        elif loss_name == 'L1Loss': # Mean Absolute Error Loss
            criterion = nn.L1Loss()
        elif loss_name == 'SmoothL1Loss': # Huber Loss
            criterion = nn.SmoothL1Loss()
        else:
            raise Exception("Loss not found")
        return criterion
        
    def custom_optimizer(self, optimizer_name, learning_rate):
        """
        This function defines the optimizer

        Args:
            optimizer_name (str): name of the optimizer
            learning_rate (float): learning rate of the optimizer
        
        Returns:
            optimizer (optim): optimizer
        """
        if optimizer_name == 'Adam':
            optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        elif optimizer_name == 'Adam_decay':
            optimizer = optim.Adam(self.model.parameters(), lr=learning_rate, weight_decay=0.0001)
        elif optimizer_name == 'SGD':
            optimizer = optim.SGD(self.model.parameters(), lr=learning_rate)
        elif optimizer_name == 'RMSprop':
            optimizer = optim.RMSprop(self.model.parameters(), lr=learning_rate)
        elif optimizer_name == 'LBFGS':
            optimizer = optim.LBFGS(self.model.parameters(), lr=learning_rate, line_search_fn='strong_wolfe')
        else:
            raise Exception("Optimizer not found")
        return optimizer
        
    def custom_learning_rate(self, lr_name): # Choose between "StepLR", "MultiStepLR", "ExponentialLR", "ReduceLROnPlateau
        """
        This function defines the learning rate scheduler

        Args:
            lr_name (str): name of the learning rate scheduler

        Returns:
            scheduler (optim): learning rate scheduler
        """
        if lr_name == 'StepLR':
            scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size=1000, gamma=0.1)
        elif lr_name == 'MultiStepLR':
            scheduler = optim.lr_scheduler.MultiStepLR(self.optimizer, milestones=[1000,10000], gamma=0.1)
        elif lr_name == 'ExponentialLR':
            scheduler = optim.lr_scheduler.ExponentialLR(self.optimizer, gamma=0.99)
        elif lr_name == 'ReduceLROnPlateau':
            scheduler = optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, mode='min', factor=0.1, patience=10, verbose=True)
        elif lr_name == 'No_scheduler':
            scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size=1, gamma=1)
        else:
            raise Exception("Learning rate not found")
        return scheduler
        
    def custom_weight_loss_updated(self, weight_data, weight_dt, weight_pinn, loss_data, loss_dt, pinn_loss, epoch):
        if self.cfg.nn.update_weight_method=="Static":
            return weight_data, weight_dt, weight_pinn
        elif self.cfg.nn.update_weight_method=="ReLoBRaLo":
            return self.relobralo_loss(loss_data, loss_dt, pinn_loss, weight_data, weight_dt, weight_pinn)
        elif self.cfg.nn.update_weight_method=="Dynamic":
            alpha_max = torch.tensor(0.2)
            epochs_to_tenfold = 50 if self.cfg.nn.optimizer == "LBFGS" else 1000
            epoch_factor = torch.tensor(10.0 ** (epoch / epochs_to_tenfold))
            weight_dt = torch.min(alpha_max, weight_dt * epoch_factor)
            weight_pinn = torch.min(alpha_max, weight_pinn * epoch_factor)
            return weight_data, weight_dt, weight_pinn
        elif self.cfg.nn.update_weight_method=="Sam":
            return weight_data, weight_dt, weight_pinn
        else:
            raise Exception("Weight update method not found")

    def custom_weight_loss_updated2(self, weight_data, weight_dt, weight_pinn, weight_pinn_ic, loss_data, loss_dt, loss_pinn, loss_pinn_ic, epoch):
        if self.cfg.nn.update_weight_method=="Static":
            return weight_data, weight_dt, weight_pinn, weight_pinn_ic
        elif self.cfg.nn.update_weight_method=="ReLoBRaLo": #SOS needs fix if this methos is adapted everywhere
            result = self.relobralo_loss(loss_data, loss_dt, loss_pinn, weight_data, weight_dt, weight_pinn)
            result.append(weight_pinn_ic)
            return result
        elif self.cfg.nn.update_weight_method=="Dynamic":
            alpha_max = torch.tensor(0.2)
            epochs_to_tenfold = 50 if self.cfg.nn.optimizer == "LBFGS" else 1000
            epoch_factor = torch.tensor(10.0 ** (epoch / epochs_to_tenfold))
            weight_dt = torch.min(alpha_max, weight_dt * epoch_factor)
            weight_pinn = torch.min(alpha_max, weight_pinn * epoch_factor)
            weight_pinn_ic = torch.min(alpha_max, weight_pinn_ic * epoch_factor)
            return weight_data, weight_dt, weight_pinn, weight_pinn_ic
        elif self.cfg.nn.update_weight_method=="Sam":
            return weight_data, weight_dt, weight_pinn, weight_pinn_ic
        elif self.cfg.nn.update_weight_method=="NTK":
            raise Exception("Not Implemented")
        ########### Add more methods KAN here ###########
        else:
            raise Exception("Weight update method not found")

    def weight_init(self,module, init_name):
        """
        This function initializes the weights of the neural network model
        
        Args:
            module (Net): neural network model
            init_name (str): name of the initialization method
        """
        for m in module.modules():
            if type(m) == nn.Linear:
                if init_name == 'xavier_normal':
                    nn.init.xavier_normal_(m.weight)
                elif init_name == 'xavier_uniform':
                    nn.init.xavier_uniform_(m.weight)
                elif init_name == 'kaiming_normal':
                    nn.init.kaiming_normal_(m.weight)
                elif init_name == 'kaiming_uniform':
                    nn.init.kaiming_uniform_(m.weight)
                elif init_name == 'normal':
                   pass
                else:
                    raise Exception("Initialization not found")
        return
        

    def test(self, x_test):
        """
        This function tests the neural network model

        Args:
            x_test (torch.Tensor): input data

        Returns:
            y_pred (torch.Tensor): predicted output data
        """
        self.model.eval()
        with torch.no_grad():
            y_pred = self.forward_pass(x_test)
        return y_pred

    def plot(self, x_train, y_train, var=0):
        """
        This function plots the data

        Args:
            x_train (torch.Tensor): input data
            y_train (torch.Tensor): output data
            var (int): variable to plot
        """
        y_pred = self.test(x_train)
        x_train = x_train[:,0].cpu().detach().numpy() # x is the time
        y_train = y_train[:,var].cpu().detach().numpy() 
        y_pred = y_pred[:,var].cpu().detach().numpy()
        plt.figure()
        plt.plot(x_train, y_train, 'ro', label='Original data')
        plt.plot(x_train, y_pred, 'kx-', label='Fitted line')
        plt.show()
        return
    
    def plot_all(self, x_train, y_train):
        """
        This function plots all the data in pairs

        Args:
            x_train (torch.Tensor): input data
            y_train (torch.Tensor): output data
        """
        y_pred = self.test(x_train)
        x_train = x_train[:,0].cpu().detach().numpy()
        y_train = y_train.cpu().detach().numpy()
        y_pred = y_pred.cpu().detach().numpy()
        plt.figure(figsize=(10, 5))  # Create a figure with a specific size
        for i in range(y_train.shape[1]):
            plt.subplot(1, 2, i % 2 + 1)  # Create subplots, alternating between two columns
            plt.plot(x_train, y_train[:, i], 'ro', label='Original data')
            plt.plot(x_train, y_pred[:, i], 'kx-', label='Fitted line')
            plt.legend()
            if i % 2 != 0:
                plt.show()  # Show the plot after every two iterations
        return
    
    def plot_all_dt(self, x_train, y_train):
        """
        This function plots the derivative of all the data in pairs
        """
        y_pred = self.test(x_train) # Predict the output data
        x_train = x_train[:,0].cpu().detach().numpy() # Keep only the time column
        dt = self.calculate_from_ode(y_train)  # Calculate the derivative of the output data
        dt_pred = self.calculate_from_ode(y_pred) # Calculate the derivative of the predicted output data
        dt = dt.cpu().detach().numpy()
        dt_pred = dt_pred.cpu().detach().numpy()
        plt.figure(figsize=(10, 5))
        for i in range(y_train.shape[1]):
            plt.subplot(1, 2, i % 2 + 1)
            plt.plot(x_train, dt[:, i], 'ro', label='Original data')
            plt.plot(x_train, dt_pred[:, i], 'kx-', label='Fitted line')
            plt.legend()
            if i % 2 != 0:
                plt.show()
        return
    
    def forward_nn(self, time, no_time):
        """
        This function calculates the output of the neural network model, input is given as time and the other input columns
        """
        x_train = torch.cat((time, no_time), 1)
        y_pred = self.model.forward(x_train)
        y_pred = self.detransform_output(y_pred) # detransform the output of the model in case it is needed for the derivative

        if self.cfg.nn.type == "PinnA":
            if self.cfg.dataset.transform_input == "None":
                return no_time + y_pred*time
            minus = self.data_loader.minus_input.clone().detach()
            divide = self.data_loader.divide_input.clone().detach()
            if self.cfg.dataset.transform_input == "MinMax2":
                div = nn.Parameter(torch.tensor(2.0), requires_grad=False).to(self.device)
                plus = nn.Parameter(torch.tensor(1.0), requires_grad=False).to(self.device)
                return ((no_time + plus) * divide[1:] / div + minus[1:]) + y_pred*((time + plus) * divide[0] / div + minus[0])
            return (no_time* divide[1:] + minus[1:]) + y_pred*(time* divide[0] + minus[0])
        if self.cfg.nn.type == "PinnB":
            return no_time + y_pred
        if self.cfg.nn.type in ["DynamicNN", "PinnAA", "ResNet"]:
            return y_pred
        else:
            raise Exception('Enter valid NN type! (zeroth_order or first_order')
        
    def forward_nn2(self, time, no_time):
        """
        This function calculates the output of the neural network model, input is given as time and the other input columns
        """
        x_train = torch.cat((time, no_time), 1)
        y_pred = self.model.forward(x_train)
        y_pred = self.detransform_output(y_pred)
        if self.cfg.nn.type == "PinnA":
            return no_time + y_pred*time
        if self.cfg.nn.type == "PinnB":
            return no_time + y_pred
        if self.cfg.nn.type in ["DynamicNN", "PinnAA", "ResNet", "KAN"]:
            return y_pred
        else:
            raise Exception('Enter valid NN type! (zeroth_order or first_order')
    
    def forward_pass(self, x_train):
        """
        This function calculates the output of the neural network model, input is given as time and the other input columns
        """
        time = x_train[:,0].unsqueeze(1) # get the time column
        no_time = x_train[:,1:]
        y_pred = self.model.forward(x_train)
        
        if self.cfg.nn.type == "PinnA":
            if self.cfg.dataset.transform_input == "None":
                return no_time + y_pred*time
            minus = self.data_loader.minus_input.clone().detach().to(self.device)
            divide = self.data_loader.divide_input.clone().detach().to(self.device)
            if self.cfg.dataset.transform_input == "MinMax2":
                div = nn.Parameter(torch.tensor(2.0), requires_grad=False)
                plus = nn.Parameter(torch.tensor(1.0), requires_grad=False)
                return ((no_time + plus) * divide[1:] / div + minus[1:]) + y_pred*((time + plus) * divide[0] / div + minus[0])
            return (no_time* divide[1:] + minus[1:]) + y_pred*(time* divide[0] + minus[0])
        if self.cfg.nn.type == "PinnB":
            return no_time + y_pred
        if self.cfg.nn.type in ["DynamicNN", "PinnAA", "ResNet"]:
            return y_pred
        else:
            raise Exception('Enter valid NN type! (zeroth_order or first_order')
            
    def forward_pass2(self, x_train):
        """
        This function calculates the output of the neural network model, input is given as time and the other input columns
        """
        time = x_train[:,0].unsqueeze(1) # get the time column
        no_time = x_train[:,1:]
        y_pred = self.model.forward(x_train)
        y_pred = self.detransform_output(y_pred)
        if self.cfg.nn.type == "PinnA":
            return no_time + y_pred*time
        if self.cfg.nn.type == "PinnB":
            return no_time + y_pred
        if self.cfg.nn.type in ["DynamicNN", "PinnAA", "ResNet","KAN"]:
            return y_pred
        else:
            raise Exception('Enter valid NN type! (zeroth_order or first_order')
        
    def derivative(self, y, t):
        """
        This function calculates the derivative of the model at d_y/d_t
        y1-y0 = dy_dt * (t1-t0), where t1-t0 = dt = 0.1001 for 1000 time intervals
        in order to crosscheck the results, I will calculate the derivative of the output of the model
        and compare it with the derivative of the output of the model
        """
        dy = torch.autograd.grad(y, t, grad_outputs = torch.ones_like(y), create_graph=True, retain_graph=True)[0]
        return dy

    def calculate_autograd2(self, x_train):
        """
        This function calculates the output of the neural network model and the derivative of the output using the derivative function
        """
        time = x_train[:,0].unsqueeze(1) # get the time column
        no_time = x_train[:,1:] # get the input columns
        y = self.forward_nn(time=time, no_time = no_time)
        u = []
        for i in range(y.shape[1]):
            u.append(self.derivative(y[:,i], time))
        u_all = torch.cat(u, 1)
        return y, u_all
    
    def calculate_autograd22(self, x_train):
        """
        This function calculates the output of the neural network model and the derivative of the output using the derivative function
        """
        time = x_train[:,0].unsqueeze(1) # get the time column

        no_time = x_train[:,1:] # get the input columns
        y = self.forward_nn2(time=time, no_time = no_time)
        u = []
        for i in range(y.shape[1]):
            u.append(self.derivative(y[:,i], time))
        u_all = torch.cat(u, 1)
        return y, u_all
    
    def calculate_autograd(self, x_train):
        """
        This function calculates the output of the neural network model and the derivative of the output 
        """
        time = x_train[:,0].unsqueeze(1) # get the time column SOSOSO check if x_train[1:,0] is required 
        no_time = x_train[:,1:] # get the input columns
        y, dy_dt = torch.autograd.functional.jvp( # calculate the jacobian vector product
            func=lambda t: self.forward_nn(time=t, no_time = no_time), inputs=time ,v=torch.ones(time.shape).to(self.device), create_graph=True, retain_graph=True)
        return y, dy_dt
    
    def calculate_from_ode(self, output):
        """
        This function calculates the output(dy/dt) of the synchronous machine model for the given input y
        """
        if self.cfg.modelling_method:
            y_processed = self.solver.synchronous_machine_equations(0, output.split(split_size=1, dim=1))
        else:
            y_processed = self.solver.synchronous_machine_equations_v2(0, output.split(split_size=1, dim=1))
        for i in range(len(y_processed)):
            if type(y_processed[i]) == int:
                value = y_processed[i]
                y_processed[i] = torch.tensor(value).repeat(output.shape[0]).unsqueeze(1).to(self.device)
        
        y_processed = torch.cat(y_processed, 1)

        return y_processed

    def calculate_point_loss(self, x_train, y_train):
        """
        This function calculates the pinn loss either for collocation points or for data points
        """
        #autograd
        #y_hat, dy_dt = self.calculate_autograd(x_train)
        y_hat, dy_dt = self.calculate_autograd2(x_train)
        #ode
        if y_train is None:
            y_hat_dtf = self.detransform_output(y_hat)
            y_processed = self.calculate_from_ode(y_hat_dtf)
        else:
            y_processed = self.calculate_from_ode(y_train)

        loss_point = self.criterion(dy_dt , y_processed)
        
        return loss_point
    
    ##To be continued
    
    # def compute_ntk_jacrev(model, x,res=0):
    #     """
    #     Compute the Neural Tangent Kernel (NTK) using torch.func.jacrev.

    #     Args:
    #         model: The neural network model (PINN).
    #         x: Input data of shape (n_samples, input_dim).

    #     Returns:
    #         ntk: The NTK matrix of shape (n_samples, n_samples).
    #     """
        
    #     # Get model parameters as a dictionary {name: param}
    #     params = {name: param for name, param in model.named_parameters()}

    #     # Define a function that computes the model output with respect to the input and parameters
    #     # def model_fn(params, x):
    #     #     return func.functional_call(model, params, (x,))
        
    #     def model_fn(params, x):
    #         x.requires_grad = True
    #         # Call the entire model and handle residuals here
    #         pred = func.functional_call(model, params, (x,))
    #         if res == 1:
    #             delta = pred[:, 0:1]
    #             delta_t = torch.autograd.grad(delta, x, torch.ones_like(delta), retain_graph=True, create_graph=True)[0]
    #             omega = pred[:, 1:2]
    #             return delta_t - omega
    #         elif res == 2:
    #             delta = pred[:, 0:1]
    #             omega = pred[:, 1:2]
    #             V = pred[:, 2:3]
    #             omega_t = torch.autograd.grad(omega, x, torch.ones_like(omega), retain_graph=True, create_graph=True)[0]
    #             Pe = system.E * V * torch.sin(delta)
    #             Pm_t = 1.0 + 0.5 * torch.sin(10 * x)
    #             return omega_t - (Pm_t - Pe - system.D * omega) / system.M
    #         elif res == 3:
    #             delta = pred[:, 0:1]
    #             V = pred[:, 2:3]
    #             V_t = torch.autograd.grad(V, x, torch.ones_like(V), retain_graph=True, create_graph=True)[0]
    #             return V_t - (system.V_ref - V + system.k * torch.sin(delta)) / system.tau
    #         else:
    #             return pred

    #     # Compute the Jacobian of the output with respect to the parameters for the entire batch
    #     jac_fn = func.jacrev(model_fn, argnums=0)  # Derivative w.r.t. model parameters

    #     # Evaluate the Jacobian
    #     jacobians = jac_fn(params, x)

    #     # Flatten each Jacobian and concatenate them along the second dimension
    #     jac_flattened = torch.cat([jac.reshape(jac.shape[0], -1) for jac in jacobians.values()], dim=1)

    #     # Compute NTK as J(x) @ J(x)^T
    #     ntk = jac_flattened @ jac_flattened.T

    #     return ntk

    def calculate_point_grad(self, x_train, y_train):
        """
        This function calculates the pinn loss either for collocation points or for data points
        """
        #autograd
        #y_hat, dy_dt = self.calculate_autograd(x_train)
    
        y_hat, dy_dt = self.calculate_autograd2(x_train) # calculate the output of the model and the derivative of the output
        #ode
        if y_train is None: # collocation points
            y_hat_dtf = self.detransform_output(y_hat) # detransform the output of the model in case it is needed
            y_processed = self.calculate_from_ode(y_hat_dtf)
            return dy_dt , y_processed
        else:
            y_processed = self.calculate_from_ode(y_train) # data points
            return y_hat, dy_dt , y_processed
    
    def calculate_point_grad2(self, x_train, y_train):
        """
        This function calculates the pinn loss either for collocation points or for data points
        """
        #autograd
        #y_hat, dy_dt = self.calculate_autograd(x_train)
    
        y_hat, dy_dt = self.calculate_autograd22(x_train) # calculate the output of the model and the derivative of the output
        #ode
        if y_train is None: # collocation points
            y_processed = self.calculate_from_ode(y_hat)
            return dy_dt , y_processed
        else:
            y_processed = self.calculate_from_ode(y_train) # data points
            return y_hat, dy_dt , y_processed
        
    def folder_name_f(self,weight_data, weight_dt, weight_pinn):
        if weight_dt == 0 and weight_pinn == 0:
            folder_name = "data"
        elif weight_data == 0 and weight_dt > 0 and weight_pinn > 0:
            folder_name = "dt_pinn"
        elif weight_data > 0 and weight_dt > 0 and weight_pinn > 0:
            folder_name = "data_dt_pinn"
        elif weight_data > 0 and weight_dt > 0 and weight_pinn == 0:
            folder_name = "data_dt"
        elif weight_data == 0 and weight_dt == 0 and weight_pinn > 0:
            folder_name = "pinn"
        else:
            raise Exception("Folder name not found")
        return folder_name
    
    def folder_name_f2(self,weight_data, weight_dt, weight_pinn, weight_pinn_ic):
        if weight_dt == 0 and weight_pinn == 0:
            if weight_pinn_ic==0 and weight_data>0: #only ground truth
                folder_name = "data"
            if weight_pinn_ic>0 and weight_data>0: # ground truth and ic
                folder_name = "data_ic"
        elif weight_data == 0 and weight_dt > 0 and weight_pinn > 0 and weight_pinn_ic==0: # only physics loss
            folder_name = "dt_pinn"
        elif weight_data > 0 and weight_dt > 0 and weight_pinn > 0: 
            if weight_pinn_ic==0:
                folder_name = "data_dt_pinn" # no ic 
            else:
                folder_name = "data_dt_pinn_ic" #all the losses
        elif weight_data > 0 and weight_dt > 0 and weight_pinn == 0 and weight_pinn_ic==0: # only ground truth losses
            folder_name = "data_dt"
        elif weight_data == 0 and weight_dt == 0 and weight_pinn > 0:
            if weight_pinn_ic==0:
                folder_name = "pinn" # only col loss without ic
            else:
                folder_name = "pinn_ic" # col loss with ic
        else:
            raise Exception("Folder name not found")
        return folder_name
    
    def define_train_val_data(self, num_of_data, num_of_skip_data_points, num_of_col_points, num_of_skip_val_points):
        """
        This function defines the training data
        Initially restrict the volume of data to both normal and collocation data
        Then sample from the data points and collocation points based on the given step size
        """
        num_of_col_data = num_of_data if num_of_data < self.data_loader.x_train_col.shape[0] else self.data_loader.x_train_col.shape[0] # max number of collocation points
        num_of_data = num_of_data if num_of_data < self.data_loader.x_train.shape[0] else self.data_loader.x_train.shape[0] # max number of data points
        x_train = self.data_loader.x_train[:num_of_data: num_of_skip_data_points].clone().detach().to(self.device).requires_grad_(True) # training data after skipping points
        y_train = self.data_loader.y_train[:num_of_data: num_of_skip_data_points].clone().detach().to(self.device)
        x_train_col = self.data_loader.x_train_col[:num_of_col_data: num_of_col_points].clone().detach().to(self.device).requires_grad_(True) # traininng collaction points after skipping points
        x_val = self.data_loader.x_val[:: num_of_skip_val_points].to(self.device) # validation data without skipping points
        y_val = self.data_loader.y_val[:: num_of_skip_val_points].to(self.device)
        self.training_shape = x_train.shape[0] # number of final training data points
        self.training_col_shape = x_train_col.shape[0] # number of final training collocation points
        self.validation_shape = x_val.shape[0] # number of final validation data points
        return x_train, y_train, x_train_col, x_val, y_val
    
    def define_train_val_data2(self, perc_of_data, perc_of_col_data, num_of_skip_data_points, num_of_col_points, num_of_skip_val_points):
        """
        This function defines the training data
        Initially restrict the volume of data to both normal and collocation data
        Then sample from the data points and collocation points based on the given step size
        """
        perc_of_data = 1 if perc_of_data > 1 else perc_of_data # max percentage of data points
        perc_of_col_data = 1 if perc_of_col_data > 1 else perc_of_col_data # max percentage of collocation points
        num_of_data = int(perc_of_data * self.data_loader.x_train.shape[0]) # max number of data points
        num_of_col_data = int(perc_of_col_data * self.data_loader.x_train_col.shape[0]) # max number of collocation points
        x_train = self.data_loader.x_train[:num_of_data: num_of_skip_data_points].clone().detach().to(self.device).requires_grad_(True) # training data after skipping points
        y_train = self.data_loader.y_train[:num_of_data: num_of_skip_data_points].clone().detach().to(self.device)
        x_train_col = self.data_loader.x_train_col[:num_of_col_data: num_of_col_points].clone().detach().to(self.device).requires_grad_(True) # traininng collaction points after skipping points
        x_train_col0 = self.data_loader.x_train_col[self.data_loader.x_train_col[:,0]==0].clone().detach().to(self.device).requires_grad_(True) # ic traininng collaction points 
        x_train_col0 = x_train_col0[:num_of_col_data: num_of_col_points].clone().detach().to(self.device).requires_grad_(True) # ic traininng collaction points after skipping points
        y_train_col0 = x_train_col0[:,1:].clone().detach().to(self.device) # ic training collocation points (when time is 0)
        x_val = self.data_loader.x_val[:: num_of_skip_val_points].to(self.device) # validation data without skipping points
        y_val = self.data_loader.y_val[:: num_of_skip_val_points].to(self.device)
        self.training_shape = x_train.shape[0] # number of final training data points
        self.training_col_shape = x_train_col.shape[0] # number of final training collocation points
        self.training_col_shape0 = x_train_col0.shape[0] # number of final training ic collocation points
        self.validation_shape = x_val.shape[0] # number of final validation data points
        return x_train, y_train, x_train_col, x_train_col0, y_train_col0 , x_val, y_val

    def transform_input(self, input):
        """
        This function transforms the input data
        """
        flag = "Input" # Flag to transform the input data
        if self.cfg.dataset.transform_input != "None":
            if self.cfg.dataset.transform_input == "Std":
                print("Standardizing input data for training")
            elif self.cfg.dataset.transform_input == "MinMax":
                print("Normalizing input data for training")
            elif self.cfg.dataset.transform_input == "MinMax2":
                print("Normalizing2 input data for training")
                flag = "Input2"
            else:
                raise Exception("Transformation not found")
            input = self.data_loader.transform_data(input,flag)
            return input.clone().detach().to(self.device).requires_grad_(True)
        else: 
            return input
        
    def detransform_input(self, input):
        """
        This function detransforms the input data
        """
        flag = "Input"
        if self.cfg.dataset.transform_input != "None":
            if self.cfg.dataset.transform_input == "Std":
                print("Unstandardizing input data ")
            elif self.cfg.dataset.transform_input == "MinMax":
                print("Unnormalizing input data")
            elif self.cfg.dataset.transform_input == "MinMax2":
                print("Unnormalizing input data")
                flag = "Input2"
            else:
                raise Exception("Transformation not found")
            input = self.data_loader.detransform_data(input,flag)
        return input.clone().detach().to(self.device).requires_grad_(True)

    def transform_output(self, output):
        """
        This function transforms the output data
        """
        flag = "Output"
        if self.cfg.dataset.transform_output != "None":
            if self.cfg.dataset.transform_output == "Std":
                pass
                #print("Standardizing output data for training")
            elif self.cfg.dataset.transform_output == "MinMax":
                pass
                #print("Normalizing output data for training")
            elif self.cfg.dataset.transform_output == "MinMax2":
                flag = "Output2"
            else:
                raise Exception("Transformation not found")
            output = self.data_loader.transform_data(output,flag)
        return output
    
    def detransform_output(self, output):
        """
        This function detransforms the output data
        """
        flag = "Output"
        if self.cfg.dataset.transform_output != "None":
            if self.cfg.dataset.transform_output == "Std":
                pass
                #print("Unstandardizing output data for testing")
            elif self.cfg.dataset.transform_output == "MinMax":
                pass
                #print("Unnormalizing output data for testing")
            elif self.cfg.dataset.transform_output == "MinMax2":
                flag = "Output2"
            else:
                raise Exception("Transformation not found")
            output = self.data_loader.detransform_data(output,flag)
        return output
    
    def pinn_train2(self, weight_data, weight_dt, weight_pinn, num_of_data, num_of_skip_data_points, num_of_skip_col_points, num_of_skip_val_points, run=None):
        """
        This function trains the neural network model

        Args:
            x_train (torch.Tensor): input data
            y_train (torch.Tensor): output data
            num_epochs (int): number of epochs
        """
        x_train, y_train, x_train_col, x_val, y_val = self.define_train_val_data(num_of_data, num_of_skip_data_points, num_of_skip_col_points, num_of_skip_val_points) # define the training and validation data
        if self.cfg.nn.type == "PinnA":
            y_train = y_train[x_train[:,0]!=0].clone().detach().to(self.device) # remove the 0 time  from the output data
            x_train = x_train[x_train[:,0]!=0].clone().detach().to(self.device).requires_grad_(True) # remove the 0 time  from the input data
            x_train_col = x_train_col[x_train_col[:,0]!=0].clone().detach().to(self.device).requires_grad_(True) # remove the 0 time from the input data
        #x_train_old = x_train.clone().detach().to(self.device).requires_grad_(True)
        #x_train_col_old = x_train_col.clone().detach().to(self.device).requires_grad_(True)
        x_train = self.transform_input(x_train) # transform the input data according to the respective chosen input_transform method 
        x_train_col = self.transform_input(x_train_col) # transform the input data according to the respective chosen input_transform method
        x_val = self.transform_input(x_val) # transform the input data according to the respective chosen input_transform method
        y_train_tf = self.transform_output(y_train) # transform the output data according to the respective chosen output_transform method
        print("Number of labeled training data: ", x_train.shape[0], " number of collocation points: ", x_train_col.shape[0], " number of validation data: ", x_val.shape[0])
        folder_name=self.folder_name_f(weight_data,weight_dt,weight_pinn)
        os.makedirs(os.path.join(self.cfg.dirs.model_dir, folder_name),exist_ok=True)

        if self.cfg.nn.update_weight_method=="Sam":
            epsilon = 1e-8
            soft_attention_weights = torch.nn.Parameter(torch.tensor([weight_data,weight_dt,weight_pinn], dtype=torch.float32, requires_grad=True))  # Learnable attention weights for each loss
        self.weight_data = weight_data 
        self.weight_dt = weight_dt
        self.weight_pinn = weight_pinn
        # training
        self.model.train()
        for epoch in range(self.cfg.nn.num_epochs):
            # training
            self.model.train() # set the model to training mode
            def closure():
                output, dydt0, ode0 = self.calculate_point_grad(x_train, y_train) # calculate nn output and its gradient for the data points, and the ode solution for the target y_train
                dydt1, ode1 = self.calculate_point_grad(x_train_col, None) # calculate nn output gradient for the collocation points, and the ode solution for the nn output
                #output_df = self.detransform_output(output) # detransform the output of the model in case it is needed
                loss_data = self.criterion(output, y_train_tf) # calculate the data loss
                loss_dt = self.criterion(dydt0, ode0) # calculate the dt loss 
                loss_pinn = self.criterion(dydt1, ode1) # calculate the pinn loss
                #loss_total =  loss_data*torch.Tensor(self.cfg.nn.loss_data_weight).to(self.device) + loss_dt*torch.Tensor(self.cfg.nn.loss_dt_weight).to(self.device)  + loss_pinn*torch.Tensor(self.cfg.nn.loss_pinn_weight).to(self.device)
                if self.cfg.nn.update_weight_method=="Dynamic":
                    self.weight_data, self.weight_dt, self.weight_pinn = self.custom_weight_loss_updated(weight_data, weight_dt, weight_pinn, loss_data, loss_dt, loss_pinn, epoch)
                elif self.cfg.nn.update_weight_method=="Sam":
                    self.weight_data, self.weight_dt, self.weight_pinn = (soft_attention_weights[0], soft_attention_weights[1], soft_attention_weights[2])
                else: # Static or ReLoBRaLo
                    self.weight_data, self.weight_dt, self.weight_pinn = self.custom_weight_loss_updated(self.weight_data, self.weight_dt, self.weight_pinn, loss_data, loss_dt, loss_pinn, epoch)

                loss_total = loss_data * self.weight_data + loss_dt * self.weight_dt + loss_pinn * self.weight_pinn
                self.loss_dt = loss_dt # save the loss for the dt outside the closure
                self.loss_pinn = loss_pinn # same
                self.loss_data = loss_data # same
                self.loss_total = loss_total # same
                self.optimizer.zero_grad() # zero the gradients
                loss_total.backward() # backpropagate the total weighted loss
                
                return loss_total
            
            self.optimizer.step(closure) # update the weights of the model
            """
            if self.optimizer != "LBFGS":
                self.scheduler.step()
            """
                #print scheduler.get_last_lr()

            # Validation
            
            self.model.eval()
            with torch.no_grad():
                outputs = self.forward_pass(x_val)
                if self.cfg.dataset.transform_output != "None":
                    outputs = self.detransform_output(outputs)
                loss = self.criterion(outputs, y_val)
                val_loss = loss.item() 
            if (epoch + 1) % 50 == 0:
                # log some plots to wandb
                if run is not None:
                    self.log_plot(outputs, y_val, epoch, run,x_val)

                print(f'Epoch [{epoch+1}/{self.cfg.nn.num_epochs}], Loss: {self.loss_total.item():.4f}, Loss_data: {self.loss_data.item():.4f}, Loss_dt: {self.loss_dt:.4f}, Loss_pinn: {self.loss_pinn:.4f}' ,self.scheduler.get_last_lr(), val_loss)
                if self.cfg.nn.update_weight_method=="Sam":
                # Update soft attention mechanism (optional)
                # For example, you can use gradient descent to update attention weights
                    with torch.no_grad():
                        if run is not None:
                            run.log({"soft_attention_weights 0": soft_attention_weights[0],  'epoch': epoch})
                            run.log({"soft_attention_weights grad0 ": soft_attention_weights.grad[0],  'epoch': epoch})
                            run.log({"soft_attention_weights 1": soft_attention_weights[1],  'epoch': epoch})
                            run.log({"soft_attention_weights grad1 ": soft_attention_weights.grad[1],  'epoch': epoch})
                            run.log({"soft_attention_weights 2": soft_attention_weights[2],  'epoch': epoch})
                            run.log({"soft_attention_weights grad2 ": soft_attention_weights.grad[2],  'epoch': epoch})
                            #run.log({"soft_attention_weights grad2 ": self.test_model(0,500),  'epoch': epoch})
                        soft_attention_weights += 0.1/(soft_attention_weights.grad + epsilon)
                        soft_attention_weights.grad.zero_()
            # log all the losses for the epoch to wandb 
            save_iteration = 500 if self.cfg.nn.optimizer == "LBFGS" else 10000 # 20 iterations within the optimizer ->500*20 = 10000
            if (epoch + 1) % save_iteration == 0:

                name = f"{self.cfg.model.model_flag}{self.cfg.nn.type}_{self.cfg.time}_{epoch+1}_{self.training_shape}_{self.training_col_shape}_{self.validation_shape}_{self.cfg.dataset.transform_input}_{self.cfg.dataset.transform_output}_{weight_data}_{weight_dt}_{weight_pinn}_{self.cfg.nn.update_weight_method}.pth"

                self.save_model(os.path.join(folder_name, name))
            
            if run is not None:
                run.log({"Val_loss": val_loss, 'epoch': epoch})
                run.log({"Loss": self.loss_total.item(), 'epoch': epoch})
                run.log({"Loss_data": self.loss_data.item(), 'epoch': epoch})
                run.log({"Loss_dt": self.loss_dt, 'epoch': epoch})
                run.log({"Loss_pinn": self.loss_pinn, 'epoch': epoch})
                run.log({"Weight_data": self.weight_data, 'epoch': epoch})
                run.log({"Weight_dt": self.weight_dt, 'epoch': epoch})
                run.log({"Weight_pinn": self.weight_pinn, 'epoch': epoch})

            if self.cfg.nn.early_stopping:
                self.early_stopping(val_loss, self.model)
                if self.early_stopping.early_stop:
                    print("Early stopping")
                    self.early_stopping.save_checkpoint(val_loss, self.model)
                    break
        if self.early_stopping.early_stop == True or (epoch + 1) % save_iteration != 0:
            name = f"{self.cfg.model.model_flag}{self.cfg.nn.type}_{self.cfg.time}_{epoch+1}_{self.training_shape}_{self.training_col_shape}_{self.validation_shape}_{self.cfg.dataset.transform_input}_{self.cfg.dataset.transform_output}_{weight_data}_{weight_dt}_{weight_pinn}_{self.cfg.nn.update_weight_method}.pth"
            self.save_model(os.path.join(folder_name, name))
        final_test_loss = self.test_model(0,500,None,run)
        return 
    
    def pinn_train(self, weight_data, weight_dt, weight_pinn, weight_pinn_ic, perc_of_data, perc_of_col_data, num_of_skip_data_points, num_of_skip_col_points, num_of_skip_val_points, run=None):
        """
        This function trains the neural network model

        Args:
            x_train (torch.Tensor): input data
            y_train (torch.Tensor): output data
            num_epochs (int): number of epochs
        """
        x_train, y_train, x_train_col, x_train_col0, y_train_col0, x_val, y_val = self.define_train_val_data2(perc_of_data, perc_of_col_data, num_of_skip_data_points, num_of_skip_col_points, num_of_skip_val_points) # define the training and validation data
        if self.cfg.nn.type == "PinnA":
            y_train = y_train[x_train[:,0]!=0].clone().detach().to(self.device) # remove the 0 time  from the output data
            x_train = x_train[x_train[:,0]!=0].clone().detach().to(self.device).requires_grad_(True) # remove the 0 time  from the input data
            x_train_col = x_train_col[x_train_col[:,0]!=0].clone().detach().to(self.device).requires_grad_(True) # remove the 0 time from the input data
            #keep only the columns without time

        x_train = self.transform_input(x_train) # transform the input data according to the respective chosen input_transform method 
        x_train_col = self.transform_input(x_train_col) # transform the input data according to the respective chosen input_transform method
        x_val = self.transform_input(x_val) # transform the input data according to the respective chosen input_transform method
        x_train_col0 = self.transform_input(x_train_col0) # transform the input data according to the respective chosen input_transform method

        #y_train_tf = self.transform_output(y_train) # transform the output data according to the respective chosen output_transform method
        
        print("Number of labeled training data: ", x_train.shape[0], " number of collocation points: ", x_train_col.shape[0], " number of collocation points ic: ", x_train_col0.shape[0], " number of validation data: ", x_val.shape[0])
        folder_name=self.folder_name_f2(weight_data,weight_dt,weight_pinn,weight_pinn_ic)
        os.makedirs(os.path.join(self.cfg.dirs.model_dir, folder_name),exist_ok=True)
        
        if self.cfg.nn.update_weight_method=="Sam":
            epsilon = 1e-8
            soft_attention_weights = torch.nn.Parameter(torch.tensor([weight_data,weight_dt,weight_pinn,weight_pinn_ic], dtype=torch.float32, requires_grad=True))  # Learnable attention weights for each loss
            self.weight_mask = torch.tensor([weight_data, weight_dt, weight_pinn, weight_pinn_ic], dtype=torch.float32)
            self.weight_mask = torch.where(self.weight_mask == 0, torch.tensor(0.0), torch.tensor(1.0))

        self.weight_data = weight_data 
        self.weight_dt = weight_dt
        self.weight_pinn = weight_pinn
        self.weight_pinn_ic = weight_pinn_ic
        # training
        self.model.train()
        print("getting in training")
        for epoch in range(self.cfg.nn.num_epochs):
            # training
            self.model.train() # set the model to training mode

            def closure():
                output, dydt0, ode0 = self.calculate_point_grad2(x_train, y_train) # calculate nn output and its gradient for the data points, and the ode solution for the target y_train
                dydt1, ode1 = self.calculate_point_grad2(x_train_col, None) # calculate nn output gradient for the collocation points, and the ode solution for the nn output
                output_col0 = self.forward_pass2(x_train_col0) # calculate the nn output for the collocation points with time 0
                loss_data = self.criterion(output, y_train) # calculate the data loss
                loss_dt = self.criterion(dydt0, ode0) # calculate the dt loss 
                loss_pinn = self.criterion(dydt1, ode1) # calculate the pinn loss
                loss_pinn_ic = self.criterion(output_col0, y_train_col0) # calculate the pinn loss for the initial condition
                #loss_total =  loss_data*torch.Tensor(self.cfg.nn.loss_data_weight).to(self.device) + loss_dt*torch.Tensor(self.cfg.nn.loss_dt_weight).to(self.device)  + loss_pinn*torch.Tensor(self.cfg.nn.loss_pinn_weight).to(self.device)
                # Update the weights of the loss functions
                if self.cfg.nn.update_weight_method=="Dynamic":
                    self.weight_data, self.weight_dt, self.weight_pinn, self.weight_pinn_ic = self.custom_weight_loss_updated2(weight_data, weight_dt, weight_pinn, weight_pinn_ic, loss_data, loss_dt, loss_pinn, loss_pinn_ic, epoch)
                elif self.cfg.nn.update_weight_method=="Sam":
                    self.weight_data, self.weight_dt, self.weight_pinn, self.weight_pinn_ic = (soft_attention_weights[0], soft_attention_weights[1], soft_attention_weights[2], soft_attention_weights[3])
                else: # Static or ReLoBRaLo
                    self.weight_data, self.weight_dt, self.weight_pinn, self.weight_pinn_ic = self.custom_weight_loss_updated2(self.weight_data, self.weight_dt, self.weight_pinn, self.weight_pinn_ic, loss_data, loss_dt, loss_pinn, loss_pinn_ic, epoch)

                loss_total = loss_data * self.weight_data + loss_dt * self.weight_dt + loss_pinn * self.weight_pinn + loss_pinn_ic * self.weight_pinn_ic
                self.loss_dt = loss_dt # save the loss for the dt outside the closure
                self.loss_pinn = loss_pinn # same
                self.loss_data = loss_data # same
                self.loss_pinn_ic = loss_pinn_ic # same
                self.loss_total = loss_total # same
                
                self.optimizer.zero_grad() # zero the gradients
                loss_total.backward() # backpropagate the total weighted loss
                
                return loss_total
            
            self.optimizer.step(closure) # update the weights of the model
            """
            if self.optimizer != "LBFGS":
                self.scheduler.step()
            """
                #print scheduler.get_last_lr()

            # Validation
            
            self.model.eval()
            with torch.no_grad():
                outputs = self.forward_pass2(x_val)
                loss = self.criterion(outputs, y_val)
                val_loss = loss.item() 
            if (epoch + 1) % 1 == 0:
                # log some plots to wandb
                if run is not None:
                    self.log_plot(outputs, y_val, epoch, run,x_val)

                print(f'Epoch [{epoch+1}/{self.cfg.nn.num_epochs}], Loss: {self.loss_total.item():.4f}, Loss_data: {self.loss_data.item():.4f}, Loss_dt: {self.loss_dt.item():.4f}, Loss_pinn: {self.loss_pinn.item():.4f} , Loss_pinn_ic : {self.loss_pinn_ic.item():.4f}', self.scheduler.get_last_lr(), val_loss)
                if self.cfg.nn.update_weight_method=="Sam":
                # Update soft attention mechanism (optional)
                # For example, you can use gradient descent to update attention weights
                    with torch.no_grad():
                        if run is not None:
                            run.log({"soft_attention_weights 0": soft_attention_weights[0],  'epoch': epoch})
                            run.log({"soft_attention_weights grad0 ": soft_attention_weights.grad[0],  'epoch': epoch})
                            run.log({"soft_attention_weights 1": soft_attention_weights[1],  'epoch': epoch})
                            run.log({"soft_attention_weights grad1 ": soft_attention_weights.grad[1],  'epoch': epoch})
                            run.log({"soft_attention_weights 2": soft_attention_weights[2],  'epoch': epoch})
                            run.log({"soft_attention_weights grad2 ": soft_attention_weights.grad[2],  'epoch': epoch})
                            run.log({"soft_attention_weights 3": soft_attention_weights[3],  'epoch': epoch})
                            run.log({"soft_attention_weights grad3 ": soft_attention_weights.grad[3],  'epoch': epoch})
                            #run.log({"soft_attention_weights grad2 ": self.test_model(0,500),  'epoch': epoch})
                        soft_attention_weights += self.weight_mask*0.1/(soft_attention_weights.grad + epsilon)
                        
                        soft_attention_weights.grad.zero_()
            # log all the losses for the epoch to wandb 
            save_iteration = 500 if self.cfg.nn.optimizer == "LBFGS" else 10000 # 20 iterations within the optimizer ->500*20 = 10000
            if (epoch + 1) % save_iteration == 0:

                name = f"{self.cfg.model.model_flag}{self.cfg.nn.type}_{self.cfg.time}_{epoch+1}_{self.training_shape}_{self.training_col_shape}_{self.validation_shape}_{self.cfg.dataset.transform_input}_{self.cfg.dataset.transform_output}_{weight_data}_{weight_dt}_{weight_pinn}_{weight_pinn_ic}_{self.cfg.nn.update_weight_method}.pth"

                self.save_model(os.path.join(folder_name, name))
            
            if run is not None:
                run.log({"Val_loss": val_loss, 'epoch': epoch})
                run.log({"Loss": self.loss_total.item(), 'epoch': epoch})
                run.log({"Loss_data": self.loss_data.item(), 'epoch': epoch})
                run.log({"Loss_dt": self.loss_dt, 'epoch': epoch})
                run.log({"Loss_pinn": self.loss_pinn, 'epoch': epoch})
                run.log({"Loss_pinn_ic": self.loss_pinn_ic, 'epoch': epoch})
                run.log({"Weight_data": self.weight_data, 'epoch': epoch})
                run.log({"Weight_dt": self.weight_dt, 'epoch': epoch})
                run.log({"Weight_pinn": self.weight_pinn, 'epoch': epoch})
                run.log({"Weight_pinn_ic": self.weight_pinn_ic, 'epoch': epoch})

            if self.cfg.nn.early_stopping:
                self.early_stopping(val_loss, self.model)
                if self.early_stopping.early_stop:
                    print("Early stopping")
                    self.early_stopping.save_checkpoint(val_loss, self.model)
                    break
        if self.early_stopping.early_stop == True or (epoch + 1) % save_iteration != 0:
            name = f"{self.cfg.model.model_flag}{self.cfg.nn.type}_{self.cfg.time}_{epoch+1}_{self.training_shape}_{self.training_col_shape}_{self.validation_shape}_{self.cfg.dataset.transform_input}_{self.cfg.dataset.transform_output}_{weight_data}_{weight_dt}_{weight_pinn}_{weight_pinn_ic}_{self.cfg.nn.update_weight_method}.pth"
            self.save_model(os.path.join(folder_name, name))
        self.final_name = os.path.join(folder_name, f"{self.cfg.model.model_flag}{self.cfg.nn.type}_{self.cfg.time}_{epoch+1}_{self.training_shape}_{self.training_col_shape}_{self.validation_shape}_{self.cfg.dataset.transform_input}_{self.cfg.dataset.transform_output}_{weight_data}_{weight_dt}_{weight_pinn}_{weight_pinn_ic}_{self.cfg.nn.update_weight_method}")
        total_test_loss =  self.test_model(0,500,"no detransform",run)
        return 
    
    def test_model(self, starting_traj=0, total_traj=1, flag=None, run=None):
        """
        This function tests the neural network model

        Args:
            x_train (torch.Tensor): input data
            y_train (torch.Tensor): output data
            num_epochs (int): number of epochs
        """
        total_traj = total_traj if total_traj < self.data_loader.total_test_trajectories else self.data_loader.total_test_trajectories
        sample_per_traj = int(self.data_loader.sample_per_traj)
        x_test = self.data_loader.x_test[starting_traj*sample_per_traj:(starting_traj+total_traj)*sample_per_traj].clone().detach().to(self.device).requires_grad_(True)
        y_test = self.data_loader.y_test[starting_traj*sample_per_traj:(starting_traj+total_traj)*sample_per_traj].clone().detach().to(self.device)
        if self.cfg.dataset.transform_input != "None":
            x_test = self.transform_input(x_test)
        self.model.eval()
        if flag is None:
            y_pred = self.forward_pass(x_test)
        else:
            y_pred = self.forward_pass2(x_test)
        test_loss = self.criterion(y_pred, y_test)
        print("Total test trajectories",total_traj)
        print(f'Loss: {test_loss.item():.8f}')
        test_loss_mae = self.criterion_mae(y_pred, y_test)
        print(f'MAE Loss: {test_loss_mae.item():.8f}')

        if run is not None:
            run.log({"Test loss": test_loss.item() })
            run.log({"MAE Test loss": test_loss_mae.item() })
            self.log_plot(y_pred, y_test, None, run,x_test)
        mae, rmse = self.loss_over_time(x_test, y_test, y_pred, run)
        return test_loss.item()

    def log_plot(self, output, target, epoch, run,x_test):
        #log in wandb
        starting_traj = 0
        total_traj = 3
        num_per_traj = int(self.data_loader.sample_per_traj)
        down_limit = starting_traj*num_per_traj
        upper_limit = (starting_traj+total_traj)*num_per_traj
        var_name = ["theta","omega(r/s)","E_q(pu)","E_d(pu)"]
        for var in range(output.shape[1]):
            plt.figure()
            plt.title(f"Trajectories {starting_traj} to {starting_traj+total_traj} for variable {var_name[var]}")
            # plot with x axis x_test[:,0]
            plt.plot(x_test[down_limit:upper_limit,0].detach().cpu().numpy(), output[down_limit:upper_limit,var].detach().cpu().numpy(), label="Predicted")
            plt.plot(x_test[down_limit:upper_limit,0].detach().cpu().numpy(), target[down_limit:upper_limit,var].detach().cpu().numpy(), label="True")
            plt.xlabel("Time(s)")
            plt.legend()
            image = wandb.Image(plt)
            if epoch is None:
                run.log({f"Test output {var}": image})
            else:
                run.log({f"Output {var}": image, 'epoch': epoch})
            plt.close()
        return
    
    def loss_over_time(self, x_test, y_test, y_pred, run = None):

        unique_values = torch.unique(x_test[:,0]) # get the unique values of the time
        mae = []
        rmse = []
        for value in unique_values: # for each time step
            index = torch.where(x_test[:,0] == value) # find the indexes of the time step
            # calculate the mae and rmse for each value
            y_pred_ = y_pred[index] # keep only the points at the specific time
            y_true = y_test[index] # keep only the points at the specific time
            mae_var = []
            rmse_var = []
            for i in range(y_pred_.shape[1]):
                mae_var.append(self.criterion_mae(y_pred_[i], y_true[i]).item()) # calculate the mae for each variable
                rmse_var.append(self.criterion(y_pred_[i], y_true[i]).item()) # calculate the rmse for each variable
            mae.append((mae_var))
            rmse.append((rmse_var))
        mae = np.array(mae)
        rmse = np.array(rmse)
        var_name = ["theta","omega(r/s)","E_q(pu)","E_d(pu)"]
        if run is not None:
            for i in range(y_pred_.shape[1]):
                """
                plt.figure()
                plt.title(f"MAE for variable {var_name[i]} over time")
                """
                mean_mae = np.mean(mae[:,i])
                """
                plt.plot(unique_values.detach().cpu().numpy(), mae[:,i], label=f"Mean MAE: {mean_mae}")
                plt.xlabel("Time(s)")
                plt.ylabel("MAE")
                plt.legend()
                run.log({f"MAE for variable {var_name[i]} over time": wandb.Image(plt)})
                plt.close()
                plt.figure()
                plt.title(f"RMSE for variable {var_name[i]} over time")
                """
                mean_rmse = np.mean(rmse[:,i])
                """
                plt.plot(unique_values.detach().cpu().numpy(), rmse[:,i], label=f"Mean RMSE: {mean_rmse}")
                plt.xlabel("Time(s)")
                plt.ylabel("RMSE")
                plt.legend()
                run.log({f"RMSE for variable {var_name[i]} over time": wandb.Image(plt)})
                plt.close()
                """
                #log only mean values
                run.log({f"Mean MAE for variable {var_name[i]}": mean_mae})
                run.log({f"Mean RMSE for variable {var_name[i]}": mean_rmse})
            time = unique_values.detach().cpu().numpy()
            for i in range(y_pred_.shape[1]):
                for j in range(time.shape[0]):
                    #run.log({"Maw ": loss_total.item(), 'epoch': epoch})
                    # log MAE and RMSE for each variable at time  
                
                    run.log({f"MAE for variable {var_name[i]}": mae[j,i], "Time": time[j]})
                    run.log({f"RMSE for variable {var_name[i]}": rmse[j,i], "Time": time[j]})


        #save the mae and rmse
        full_path = os.path.join(self.cfg.dirs.model_dir, self.final_name)
        np.save(full_path+"_mae.npy", mae)
        np.save(full_path+"_rmse.npy", rmse)

        return mae, rmse

    def save_model(self,name):
        """
        Save model weights to the model_dir.      

        Args:
            name (str): name of the model 
        """
        #save model to the model_dir
        model_dir = self.cfg.dirs.model_dir
        #find if there is such folder in the model_dir
        if not os.path.exists(model_dir):
            os.makedirs(model_dir)

        model_path = os.path.join(model_dir,name)
        

        model_data = {"model_state_dict":self.model.state_dict()}
        if self.cfg.dataset.transform_input != "None":
            #extend the model_data dict
            model_data["minus_input"] = self.data_loader.minus_input
            model_data["divide_input"] = self.data_loader.divide_input

        if self.cfg.dataset.transform_output != "None":
            model_data["minus_target"] = self.data_loader.minus_target
            model_data["divide_target"] = self.data_loader.divide_target
    
        torch.save(model_data, model_path)
        
        print("Model( and tf values) saved:", model_path)
        return
    
    def load_model(self,name=None):
        """
        Load neural network model weights from the model_dir.

        Args:
            name (str): name of the model
        """
        #load model from the model_dir
        model_dir = self.cfg.dirs.model_dir
        if not os.path.exists(model_dir) or len(os.listdir(model_dir))==0:
            raise Exception("No model found in the model_dir, please correct path and name of the model from the bucket first")
        if name is None:
            #find first model in the model_dir
            name=os.listdir(model_dir)[0]
            if name=='.gitkeep':
                if len(os.listdir(model_dir))==1:
                    raise Exception("No model found in the model_dir, please correct path and name of the model from the bucket first")
                name=os.listdir(model_dir)[1]
            print("load model:",name)
        
        model_path = os.path.join(model_dir, name)
        if not os.path.exists(model_path):
            print(os.path.join(model_dir,name))
            raise Exception("No model found in the model_dir, please correct path and name of the model from the bucket first")

        model_data = torch.load(model_path)
        self.model.load_state_dict(model_data['model_state_dict'])
        return None


    
