from src.functions import *
from src.params import *
from src.ode.sm_models_d import SynchronousMachineModels 
from src.ode.sm_models_d_torch import SynchronousMachineModelsTorch
from omegaconf import OmegaConf
import os
import wandb
import pickle
import numpy as np
from pyDOE import lhs

class SM_modelling():
    def __init__(self, config):
        """
        Initializes an instance of ClassName.
        Args:
            config: The configuration object containing various parameters.

        Attributes:
            config (object): The configuration object.
            model_flag (str): The model flag: SM, SM_AVR or SM_AVR_GOV.
            time (str): The time interval for the simulation.
            machine_num (int): The machine number that will be used.
            init_conditions (str): The initial conditions set 
            params_dir (str): The directory for parameters: machine, avr, gov and system.
            init_conditions_dir (str): The directory for initial conditions per machine modelling type: SM, SM_AVR and SM_AVR_GOV.
            dataset_dir (str): The directory for saving the dataset.
            torch (bool): The flag to use PyTorch for the model.

        Methods:
            define_machine_params: Define the parameters of the synchronous machine based on the machine_num.
            define_system_params: Define the system parameters of the synchronous machine.
            append_element: Appends an element to each state in the given value set by iterating over a range of values.
            append_element_set: Appends an element to each state in the given value set by iterating over a range of values.
            create_init_conditions_set: Define the various initial conditions of the synchronous machine and return a matrix with all the possible combinations.
            create_solver: Create the solver for the synchronous machine model.
            solve_sm_model: Solves the synchronous machine model for multiple initial conditions.
            save_dataset: Create and save dataset for the model.
            load_dataset: Load the dataset.
        """
        self.config = config
        self.method = config.modelling_method
        self.model_flag = config.model.model_flag
        self.time = config.time
        self.num_of_points = config.num_of_points
        self.machine_num = config.model.machine_num
        self.init_condition_bounds = config.model.init_condition_bounds
        self.sampling = config.model.sampling
        self.params_dir = config.dirs.params_dir
        self.init_conditions_dir = config.dirs.init_conditions_dir
        self.dataset_dir = config.dirs.dataset_dir
        self.torch = config.model.torch
        self.seed = None if not hasattr(config.model, 'seed') else config.model.seed


    def define_machine_params(self):
        """
        Define the parameters of the synchronous machine based on the machine_num
        and potentially the parameters of the AVR and the Governor.

        Returns:
            Tuple: A tuple containing the machine parameters, AVR parameters, and Governor parameters.
                - machine_params (dict): The parameters of the synchronous machine.
                - avr_params (dict): The parameters of the automatic voltage regulator (AVR).
                - gov_params (dict): The parameters of the governor.
        """
        machine_params_path = os.path.join(self.params_dir, "machine" + str(self.machine_num) + ".yaml") # path to the selected machine parameters
        machine_params = OmegaConf.load(machine_params_path)

        if self.model_flag=="SM_AVR" or self.model_flag=="SM_AVR_GOV": 
            avr_params_path= os.path.join(self.params_dir,"avr.yaml") #path to the automatic voltage regulator parameters
            avr_params= OmegaConf.load(avr_params_path)
        else: 
            avr_params=None

        if self.model_flag=="SM_AVR_GOV":
            gov_params_path= os.path.join(self.params_dir,"gov.yaml") #path to the governor parameters
            gov_params= OmegaConf.load(gov_params_path)
        else:
            gov_params=None

        return machine_params, avr_params, gov_params
    
    def define_system_params(self):
        """
        Define the system parameters of the synchronous machine.

        Returns:
            dict: A dictionary containing the system parameters.
        """
        system_params_path = os.path.join(self.params_dir, "system1.yaml") # path to the system parameters
        print(system_params_path)
        system_params = OmegaConf.load(system_params_path)
        return system_params
    
    def append_element(self, value_set, Value_range, num_ranges):
        """
        Appends an element to each state in the given value set by iterating over a range of values.

        Args:
            value_set (list): The list of states to which the element will be appended.
            Value_range (tuple): The range of values from which the element will be selected.
            num_ranges (int): The number of ranges to divide the Value_range into.

        Returns:
            list: A new list of states with the element appended.

        """
        new_value_set = []
        for j in range(len(value_set)):
            for i in range(num_ranges):
                value = (Value_range[0] + i * (Value_range[1] - Value_range[0]) / (num_ranges - 1) if num_ranges > 1 else Value_range[0])
                new_state = value_set[j].copy()
                new_state.extend([value])
                new_value_set.append(new_state)
        return new_value_set
    
    
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

    def create_init_conditions_set2(self):
        """
        Define the various initial conditions of the synchronous machine and return a matrix with all the possible combinations.

        Returns:
            list: A matrix with all the possible combinations of initial conditions.
        """
        if self.torch:# if using torch then use the nn_init_cond.yaml file to create collocation points init conditions
            init_conditions_path = os.path.join(self.init_conditions_dir, self.model_flag,"nn_init_cond"+str(self.init_condition_bounds)+".yaml")
        else: # 
            init_conditions_path = os.path.join(self.init_conditions_dir, self.model_flag,"init_cond"+str(self.init_condition_bounds)+".yaml")
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
        number_of_conditions = Theta_iterations * Omega_iterations * E_d_dash_iterations * E_q_dash_iterations
        if self.model_flag=="SM_AVR" or self.model_flag=="SM_AVR_GOV":
            R_F = init_conditions.R_F
            R_F_iterations = init_conditions.R_F_iterations
            V_r = init_conditions.V_r
            V_r_iterations = init_conditions.V_r_iterations
            E_fd = init_conditions.E_fd
            E_fd_iterations = init_conditions.E_fd_iterations
            number_of_conditions=number_of_conditions*R_F_iterations*V_r_iterations*E_fd_iterations
            if self.model_flag=="SM_AVR_GOV":
                P_m = init_conditions.P_m
                P_m_iterations = init_conditions.P_m_iterations
                P_sv = init_conditions.P_sv
                P_sv_iterations = init_conditions.P_sv_iterations
                number_of_conditions=number_of_conditions*P_m_iterations*P_sv_iterations
        if self.torch:
            print("Number of different initial conditions for collocation points: ", number_of_conditions)
            #wandb.log({"Number of different initial conditions for collocation points: ": number_of_conditions})
        else:
            print("Number of different initial conditions: ", number_of_conditions)
            wandb.log({"Number of different initial conditions: ": number_of_conditions})
        init_condition_table = []   
        for i in range(Theta_iterations):
            theta_f = (theta[0] + i * (theta[1] - theta[0]) / (Theta_iterations - 1) if Theta_iterations > 1 else theta[0])
            init_condition_table.append([theta_f])
        init_condition_table = self.append_element(init_condition_table, omega, Omega_iterations)
        init_condition_table = self.append_element(init_condition_table, E_d_dash, E_d_dash_iterations)
        init_condition_table = self.append_element(init_condition_table, E_q_dash, E_q_dash_iterations)
        if self.model_flag=="SM_AVR" or self.model_flag=="SM_AVR_GOV":
            init_condition_table = self.append_element(init_condition_table, R_F, R_F_iterations)
            init_condition_table = self.append_element(init_condition_table, V_r, V_r_iterations)
            init_condition_table = self.append_element(init_condition_table, E_fd, E_fd_iterations)
            if self.model_flag=="SM_AVR_GOV":
                init_condition_table = self.append_element(init_condition_table, P_m, P_m_iterations)
                init_condition_table = self.append_element(init_condition_table, P_sv, P_sv_iterations)
        return init_condition_table
    
    def create_init_conditions_set3(self):
        """
        Define the various initial conditions of the synchronous machine and return a matrix with all the possible combinations.

        Returns:
            list: A matrix with all the possible combinations of initial conditions.
        """
        if self.torch:# if using torch then use the nn_init_cond.yaml file to create collocation points init conditions
            init_conditions_path = os.path.join(self.init_conditions_dir, self.model_flag,"nn_init_cond"+str(self.init_condition_bounds)+".yaml")
        else: # 
            init_conditions_path = os.path.join(self.init_conditions_dir, self.model_flag,"init_cond"+str(self.init_condition_bounds)+".yaml")
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
        if self.torch:
            print("Number of different initial conditions for collocation points: ", number_of_conditions)
            #wandb.log({"Number of different initial conditions for collocation points: ": number_of_conditions})
        else:
            print("Number of different initial conditions: ", number_of_conditions)
            wandb.log({"Number of different initial conditions: ": number_of_conditions})
        print(set_of_values,"Set of values for init conditions")
        print(iterations,"Iterations per value")
        #wandb.log({"Set of values for init conditions: ": set_of_values})
        #wandb.log({"Iterations per value: ": iterations})
        init_condition_table = []   
        for k in range(len(set_of_values)):
            init_condition_table = self.append_element_set(init_condition_table, set_of_values[k], iterations[k])
        return init_condition_table

    def create_init_conditions_set(self):
        """
        Define the various initial conditions of the synchronous machine and return a matrix with all the possible combinations.

        Returns:
            list: A matrix with all the possible combinations of initial conditions.
        """
        init_conditions_path = os.path.join(self.init_conditions_dir, self.model_flag,"init_cond"+str(self.init_condition_bounds)+".yaml") # path to the initial conditions
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
        number_of_conditions = Theta_iterations * Omega_iterations * E_d_dash_iterations * E_q_dash_iterations
        if self.model_flag=="SM_AVR" or self.model_flag=="SM_AVR_GOV":
            R_F = init_conditions.R_F
            R_F_iterations = init_conditions.R_F_iterations
            V_r = init_conditions.V_r
            V_r_iterations = init_conditions.V_r_iterations
            E_fd = init_conditions.E_fd
            E_fd_iterations = init_conditions.E_fd_iterations
            number_of_conditions=number_of_conditions*R_F_iterations*V_r_iterations*E_fd_iterations
            if self.model_flag=="SM_AVR_GOV":
                P_m = init_conditions.P_m
                P_m_iterations = init_conditions.P_m_iterations
                P_sv = init_conditions.P_sv
                P_sv_iterations = init_conditions.P_sv_iterations
                number_of_conditions=number_of_conditions*P_m_iterations*P_sv_iterations

        print("Number of different initial conditions: ", number_of_conditions)
        wandb.log({"Number of different initial conditions: ": number_of_conditions})
        init_condition_table = []   
        for i in range(Theta_iterations):
            theta_f = (theta[0] + i * (theta[1] - theta[0]) / (Theta_iterations - 1) if Theta_iterations > 1 else theta[0])
            for j in range(Omega_iterations):
                omega_f = (omega[0] + j * (omega[1] - omega[0]) / (Omega_iterations - 1) if Omega_iterations > 1 else omega[0])
                for k in range(E_d_dash_iterations):
                    E_d_dash_f = (E_d_dash[0] + k * (E_d_dash[1] - E_d_dash[0]) / (E_d_dash_iterations - 1) if E_d_dash_iterations > 1 else E_d_dash[0])
                    for l in range(E_q_dash_iterations):
                        E_q_dash_f = (E_q_dash[0] + l * (E_q_dash[1] - E_q_dash[0]) / (E_q_dash_iterations - 1) if E_q_dash_iterations > 1 else E_q_dash[0])
                        if self.model_flag=="SM_AVR" or self.model_flag=="SM_AVR_GOV":
                            for m in range(R_F_iterations):
                                R_F_f = (R_F[0] + m * (R_F[1] - R_F[0]) / (R_F_iterations - 1) if R_F_iterations > 1 else R_F[0])
                                for n in range(V_r_iterations):
                                    V_r_f = (V_r[0] + n * (V_r[1] - V_r[0]) / (V_r_iterations - 1) if V_r_iterations > 1 else V_r[0])
                                    for o in range(E_fd_iterations):
                                        E_fd_f = (E_fd[0] + o * (E_fd[1] - E_fd[0]) / (E_fd_iterations - 1) if E_fd_iterations > 1 else E_fd[0])
                                        if self.model_flag=="SM_AVR_GOV":
                                            for p in range(P_m_iterations):
                                                P_m_f = (P_m[0] + p * (P_m[1] - P_m[0]) / (P_m_iterations - 1) if P_m_iterations > 1 else P_m[0])
                                                for q in range(P_sv_iterations):
                                                    P_sv_f = (P_sv[0] + q * (P_sv[1] - P_sv[0]) / (P_sv_iterations - 1) if P_sv_iterations > 1 else P_sv[0])
                                                    init_condition=[theta_f, omega_f, E_d_dash_f, E_q_dash_f, R_F_f, V_r_f, E_fd_f, P_m_f, P_sv_f]
                                                    init_condition_table.append(init_condition)
                                        else:
                                            init_condition=[theta_f, omega_f, E_d_dash_f, E_q_dash_f, R_F_f, V_r_f, E_fd_f]
                                            init_condition_table.append(init_condition)
                        else:
                            init_condition=[theta_f, omega_f, E_d_dash_f, E_q_dash_f]   
                            init_condition_table.append(init_condition)
        return init_condition_table
    
    
    
    def create_solver(self, machine_params, system_params):
        """
        Create the solver for the synchronous machine model.

        Args:
            machine_params (dict): Dictionary containing the parameters of the synchronous machine.
            system_params (dict): Dictionary containing the parameters of the power system.
        Returns:
            SynchronousMachineModels: The synchronous machine model solver.
        """
        solver = SynchronousMachineModels(self.time, self.num_of_points, machine_params, system_params, self.model_flag)
        return solver

    def solve_sm_model(self, machine_params, system_params, init_conditions):
        """
        Solves the synchronous machine model for multiple initial conditions.

        Args:
            machine_params (dict): Dictionary containing the parameters of the synchronous machine.
            system_params (dict): Dictionary containing the parameters of the power system.
            init_conditions (list): List of initial conditions for which the model needs to be solved.

        Returns:
            list: List of solutions for each initial condition.

        """
        solver = self.create_solver(machine_params, system_params)
        solution_all=[]
        for i in range(len(init_conditions)):
            solution = solver.solve(init_conditions[i], self.method)
            solution_all.append(solution)
        return solution_all
    
    def save_dataset(self, solution):
        """
        Create and save dataset for the model.

        Args:
            solution (list): The solution of the differential equations of the synchronous machine.

        Returns:
            list: The dataset of the synchronous machine.
        """
        dataset = []
        for i in range(len(solution)):
            r = [solution[i].t]  # append time to directory
            for j in range(len(solution[i].y)):
                r.append(solution[i].y[j])  # append the solution at each time step
            dataset.append(r)

        # check if folder exists if not create it
        if not os.path.exists(os.path.join(self.dataset_dir, self.model_flag)):
            os.makedirs(os.path.join(self.dataset_dir, self.model_flag))

        # count the number of files in the directory
        num_files = len([f for f in os.listdir(os.path.join(self.dataset_dir, self.model_flag)) if os.path.isfile(os.path.join(self.dataset_dir, self.model_flag, f))])
        print("Number of files in the directory: ", num_files)
        print(f'Saved dataset "{self.model_flag, "dataset_v" + str(num_files + 1)}".')
        wandb.log({"Dataset saved": f'Saved dataset "{self.model_flag, "dataset_v" + str(num_files + 1)}".'})
        # save the dataset as pickle in the dataset directory
        dataset_path = os.path.join(self.dataset_dir, self.model_flag, "dataset_v" + str(num_files + 1) + ".pkl")
        with open(dataset_path, 'wb') as f:
            pickle.dump(dataset, f)

        return dataset

   

    def load_dataset(self, name):
        """
        Load the dataset.

        Args:
            name (str): The name of the dataset.

        Returns:
            list: The dataset of the synchronous machine.
        """
        dataset_path = os.path.join(self.dataset_dir, self.model_flag, name)
        with open(dataset_path, 'rb') as f:
            dataset = pickle.load(f)
        return dataset

  



    