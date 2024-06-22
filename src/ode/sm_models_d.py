import numpy as np
from scipy.integrate import solve_ivp
from src.functions import set_time
import torch

class SynchronousMachineModels():
    def __init__(self, t, num_of_points, machine_params, system_params, model_flag):
        """
        Initialize the synchronous machine model with Automatic Voltage Regulator (AVR).

        Parameters:
            t (float): The current time.
            num_of_points (int): The number of points to evaluate the solution at.
            machine_params (list): The parameters of the synchronous machine.
            system_params (object): The system parameters of the synchronous machine.
            model_flag (str): The flag indicating the type of model.

        Attributes:
            t (float): Time for solving ode.
            machine_params (list): The parameters of the synchronous machine.
            system_params (object): The system parameters of the synchronous machine.
            model_flag (str): The flag indicating the type of model.
            model (str): The type of synchronous machine model.

            X_d_dash (float): The synchronous reactance in the direct axis.
            X_q_dash (float): The synchronous reactance in the quadrature axis.
            Vs (float): The stator voltage magnitude.
            theta_vs (float): The stator voltage angle.
            omega_B (float): The base angular frequency.
            H (float): The inertia constant.
            P_m (float): The mechanical power input (only if governor exists).
            D (float): The damping coefficient.
            T_q_dash (float): The transient reactance time constant (not for infinite bus model).
            X_q (float): The synchronous reactance in the quadrature axis (not for infinite bus model).
            T_d_dash (float): The transient reactance time constant (not for infinite bus model).
            X_d (float): The synchronous reactance in the direct axis (not for infinite bus model).
            R_s (float): The stator resistance (not for infinite bus model).
            E_fd (float): The field voltage (only if AVR exists).
            K_A (float): The AVR gain (only if AVR exists).
            T_A (float): The AVR time constant (only if AVR exists).
            K_E (float): The AVR excitation system gain (only if AVR exists).
            T_E (float): The AVR excitation system time constant (only if AVR exists).
            K_F (float): The AVR feedback gain (only if AVR exists).
            T_F (float): The AVR feedback time constant (only if AVR exists).
            V_ref (float): The reference voltage (only if AVR exists).
            P_c (float): The governor control signal (only if governor exists).
            R_d (float): The governor droop coefficient (only if governor exists).
            T_ch (float): The governor time constant (only if governor exists).
            T_sv (float): The governor servo time constant (only if governor exists).
            

        """
        self.t = t
        self.num_of_points = num_of_points
        self.machine_params = machine_params
        self.system_params = system_params
        self.model_flag = model_flag
        # parameters of the synchronous machine
        self.X_d_dash = machine_params[0].X_d_dash
        self.X_q_dash = machine_params[0].X_q_dash
        self.Vs       = system_params.Vs
        self.theta_vs = system_params.theta_vs
        self.omega_B  = system_params.omega_B
        self.H        = machine_params[0].H
        if not model_flag=="SM_AVR_GOV": # if governor exists, then P_m is a control input
            self.P_m  = machine_params[0].P_m
        self.D        = machine_params[0].D 

        if not (model_flag=="SM_IB"): # not infinite bus parameters
            self.T_q_dash = machine_params[0].T_q_dash
            self.X_q      = machine_params[0].Xq
            self.T_d_dash = machine_params[0].T_d_dash
            self.X_d      = machine_params[0].Xd
            self.R_s      = machine_params[0].Rs
        if not (model_flag=="SM_AVR" or model_flag=="SM_AVR_GOV"): # if automatic voltage regulator exists E_fd is a control input
            self.E_fd     = machine_params[0].E_fd

        if model_flag=="SM_IB":
            self.model= "Infinite bus model"
            #print("Infinite bus model")

        if model_flag=="SM": # 2 axis Synchronous Machine Model
            self.model= "2 axis Synchronous Machine Model"
            #print("2 axis Synchronous Machine Model")

        if model_flag=="SM_AVR":
            self.model= "2 axis Synchronous Machine Model with AVR"
            #print("2 axis Synchronous Machine Model with AVR")

        if (model_flag=="SM_AVR" or model_flag=="SM_AVR_GOV"): # automatic voltage regulator parameters
            self.K_A      = machine_params[1].KA
            self.T_A      = machine_params[1].TA
            self.K_E      = machine_params[1].KE
            self.T_E      = machine_params[1].TE
            self.K_F      = machine_params[1].KF
            self.T_F      = machine_params[1].TF
            self.V_ref    = machine_params[1].V_ref
          
        if model_flag=="SM_AVR_GOV": # governor parameters
            self.model= "2 axis Synchronous Machine Model with AVR and governor"
            #print("2 axis Synchronous Machine Model with AVR and governor")
            self.P_c      = machine_params[2].P_c
            self.R_d      = machine_params[2].R_d
            self.T_ch     = machine_params[2].T_ch
            self.T_sv     = machine_params[2].T_sv


    def calculate_currents(self, theta, E_d_dash, E_q_dash):
        """
        Calculates the currents I_d and I_q based on the given parameters.

        Parameters:
        theta (rad): The angle .
        E_d_dash (pu): The value of E_d_dash.
        E_q_dash (pu): The value of E_q_dash.

        Returns:
        tuple: A tuple containing the calculated values of I_d and I_q.
        """

        Rs=0.0
        Re=0.0
        Xep=0.0
        alpha = [[(Rs+Re), -(self.X_q_dash+Xep)], [(self.X_d_dash+Xep), (Rs+Re)]]
        #beta = [[E_d_dash - self.Vs*np.sin(theta-self.theta_vs)], [E_q_dash - self.Vs*np.cos(theta-self.theta_vs)]]
        
        inv_alpha = np.linalg.inv(alpha)
        # Calculate I_d and I_q
        if isinstance(theta, torch.Tensor):
            beta = [[E_d_dash - self.Vs*torch.sin(theta-self.theta_vs)], [E_q_dash - self.Vs*torch.cos(theta-self.theta_vs)]]
        else:
            beta = [[E_d_dash - self.Vs*np.sin(theta-self.theta_vs)], [E_q_dash - self.Vs*np.cos(theta-self.theta_vs)]]
            
        I_d= inv_alpha[0][0]*beta[0][0] + inv_alpha[0][1]*beta[1][0]
        I_q= inv_alpha[1][0]*beta[0][0] + inv_alpha[1][1]*beta[1][0]
        #if isinstance(theta, torch.Tensor):
        #    I_d= inv_alpha[0][0]*(E_d_dash - self.Vs*torch.sin(theta-self.theta_vs)) + inv_alpha[0][1]*(E_q_dash - self.Vs*torch.cos(theta-self.theta_vs))
        #    I_q= inv_alpha[1][0]*(E_d_dash - self.Vs*torch.sin(theta-self.theta_vs)) + inv_alpha[1][1]*(E_q_dash - self.Vs*torch.cos(theta-self.theta_vs))
        #else:
        #    I_d= inv_alpha[0][0]*(E_d_dash - self.Vs*np.sin(theta-self.theta_vs)) + inv_alpha[0][1]*(E_q_dash - self.Vs*np.cos(theta-self.theta_vs))
        #    I_q= inv_alpha[1][0]*(E_d_dash - self.Vs*np.sin(theta-self.theta_vs)) + inv_alpha[1][1]*(E_q_dash - self.Vs*np.cos(theta-self.theta_vs))

        #I_t = np.matmul(inv_alpha, beta)
        #I_d = I_t[0][0]
        #I_q = I_t[1][0]
        
        return I_d, I_q

    def calculate_voltages(self, theta, I_d, I_q):
        """
        Calculate the voltage V_t based on the given inputs, for AVR model

        Parameters:
        theta (rad): The angle in radians.
        I_d (pu): The d-axis current.
        I_q (pu): The q-axis current.

        Returns:
        float: The magnitude of the total voltage V_t(pu).
        """
        Re = 0.0
        Xep = 0.0
        if isinstance(theta, torch.Tensor):
            V_d = Re * I_d - Xep * I_q + self.Vs * torch.sin(theta - self.theta_vs)
            V_q = Re * I_q + Xep * I_d + self.Vs * torch.cos(theta - self.theta_vs)
            V_t = torch.sqrt(V_d ** 2 + V_q ** 2)
        else:
            V_d = Re * I_d - Xep * I_q + self.Vs * np.sin(theta - self.theta_vs)
            V_q = Re * I_q + Xep * I_d + self.Vs * np.cos(theta - self.theta_vs)
            V_t = np.sqrt(V_d ** 2 + V_q ** 2)  # equal to Vs
        return V_t
        
    def synchronous_machine_equations(self, t, x):
        """
        Calculates the derivatives of the state variables for the synchronous machine model.

        Parameters:
            t (float): The current time.
            x (list): A list of state variables, different for each model type.

        Returns:
            list: A list of derivatives, different for each model type.
        """
        if self.model_flag=="SM_IB" or self.model_flag=="SM":
            theta, omega, E_d_dash, E_q_dash = x
        if self.model_flag=="SM_AVR":
            theta, omega, E_d_dash, E_q_dash, R_F, V_r, E_fd = x
        if self.model_flag=="SM_AVR_GOV":
            theta, omega, E_d_dash, E_q_dash, R_F, V_r, E_fd, P_m, P_sv = x

        # Calculate currents from algebraic equations
        I_d, I_q = self.calculate_currents(theta, E_d_dash, E_q_dash)

        if (self.model_flag=="SM_AVR" or self.model_flag=="SM_AVR_GOV"): # calculate V_t from algebraic equations
            V_t = self.calculate_voltages(theta, I_d, I_q)
        
        # Calculate theta derivative
        dtheta_dt = omega
        
        # Calculate omega derivative
        if self.model_flag=="SM_AVR_GOV": # calculate omega derivative from algebraic equations
            domega_dt = (self.omega_B / (2 * self.H)) * (P_m - E_d_dash * I_d - E_q_dash * I_q - (self.X_q_dash - self.X_d_dash) * I_q * I_d - self.D * omega)
        else:
            domega_dt = (self.omega_B / (2 * self.H)) * (self.P_m - E_d_dash * I_d - E_q_dash * I_q - (self.X_q_dash - self.X_d_dash) * I_q * I_d - self.D * omega)
        
        # Calculate E_dash derivatives
        if self.model_flag=="SM_IB":
            dE_q_dash_dt = 0
            dE_d_dash_dt = 0
            return [dtheta_dt, domega_dt, dE_d_dash_dt, dE_q_dash_dt]
        
        if self.model_flag=="SM":
            dE_q_dash_dt = (1 / self.T_d_dash) * (- E_q_dash - I_d * (self.X_d - self.X_d_dash) + self.E_fd)
            dE_d_dash_dt = (1 / self.T_q_dash) * (- E_d_dash + I_q * (self.X_q - self.X_q_dash))
            return [dtheta_dt, domega_dt, dE_d_dash_dt, dE_q_dash_dt]

        
        # Automatic Voltage Regulator (AVR) dynamics 4.46-4.48
        # Exciter and AVR equations
        if (self.model_flag=="SM_AVR" or self.model_flag=="SM_AVR_GOV"):
            dE_q_dash_dt = (1 / self.T_d_dash) * (- E_q_dash - I_d * (self.X_d - self.X_d_dash) + E_fd)
            dE_d_dash_dt = (1 / self.T_q_dash) * (- E_d_dash + I_q * (self.X_q - self.X_q_dash))
            dR_F_dt      = (1 / self.T_F) * (-R_F + (self.K_F / self.T_F) * E_fd)
            dV_r_dt      = (1 / self.T_A) * (-V_r + (self.K_A * R_F) - (self.K_A * self.K_F / self.T_F) * E_fd + self.K_A * (self.V_ref - V_t))
            dE_fd_dt     = (1 / self.T_E) * (-(self.K_E + 0.098 * np.e**(E_fd*0.55)) * E_fd + V_r)
            
            if self.model_flag=="SM_AVR_GOV":        # Governor equations # recheck it after the meeting
                dP_m_dt  = (1 / self.T_ch) * (-P_m  + P_sv) # 4.110  from dynamics dT_m_dt = - P_m / (2 * H) + P_sv 4.100 + check draw.io
                dP_sv_dt = (1 / self.T_sv) * (-P_sv + self.P_c - (1/self.R_d) *(omega/self.omega_B))
                return [dtheta_dt, domega_dt, dE_d_dash_dt, dE_q_dash_dt, dR_F_dt, dV_r_dt, dE_fd_dt, dP_m_dt, dP_sv_dt]#4.116 recheck it after the meeting
            return [dtheta_dt, domega_dt, dE_d_dash_dt, dE_q_dash_dt, dR_F_dt, dV_r_dt, dE_fd_dt]

    def synchronous_machine_equations_v2(self, t, x):
        """
        Calculates the derivatives of the state variables for the synchronous machine model.

        Parameters:
            t (float): The current time.
            x (list): A list of state variables, different for each model type.

        Returns:
            list: A list of derivatives, different for each model type.
        """
        if self.model_flag=="SM_IB" or self.model_flag=="SM":
            theta, omega, E_d_dash, E_q_dash = x #SOSOSOS in that case devde by self.omega_B is needed cause the disturbance is in omega
        if self.model_flag=="SM_AVR":
            theta, omega, E_d_dash, E_q_dash, R_F, V_r, E_fd = x
        if self.model_flag=="SM_AVR_GOV":
            theta, omega, E_d_dash, E_q_dash, R_F, V_r, E_fd, P_m, P_sv = x

        # Calculate currents from algebraic equations
        I_d, I_q = self.calculate_currents(theta, E_d_dash, E_q_dash)

        if (self.model_flag=="SM_AVR" or self.model_flag=="SM_AVR_GOV"): # calculate V_t from algebraic equations
            V_t = self.calculate_voltages(theta, I_d, I_q)
        
        # Calculate theta derivative
        dtheta_dt = omega * self.omega_B
        
        # Calculate omega derivative
        if self.model_flag=="SM_AVR_GOV": # calculate omega derivative from algebraic equations
            domega_dt = (1/ (2 * self.H)) * (P_m - E_d_dash * I_d - E_q_dash * I_q - (self.X_q_dash - self.X_d_dash) * I_q * I_d - self.D * omega * self.omega_B)
        else:
            domega_dt = (1 / (2 * self.H)) * (self.P_m - E_d_dash * I_d - E_q_dash * I_q - (self.X_q_dash - self.X_d_dash) * I_q * I_d - self.D * omega * self.omega_B)
        
        # Calculate E_dash derivatives
        if self.model_flag=="SM_IB":
            dE_q_dash_dt = 0
            dE_d_dash_dt = 0
            return [dtheta_dt, domega_dt, dE_d_dash_dt, dE_q_dash_dt]
        
        if self.model_flag=="SM":
            dE_q_dash_dt = (1 / self.T_d_dash) * (- E_q_dash - I_d * (self.X_d - self.X_d_dash) + self.E_fd)
            dE_d_dash_dt = (1 / self.T_q_dash) * (- E_d_dash + I_q * (self.X_q - self.X_q_dash))
            return [dtheta_dt, domega_dt, dE_d_dash_dt, dE_q_dash_dt]

        
        # Automatic Voltage Regulator (AVR) dynamics 4.46-4.48
        # Exciter and AVR equations
        if (self.model_flag=="SM_AVR" or self.model_flag=="SM_AVR_GOV"):
            dE_q_dash_dt = (1 / self.T_d_dash) * (- E_q_dash - I_d * (self.X_d - self.X_d_dash) + E_fd)
            dE_d_dash_dt = (1 / self.T_q_dash) * (- E_d_dash + I_q * (self.X_q - self.X_q_dash))
            dR_F_dt      = (1 / self.T_F) * (-R_F + (self.K_F / self.T_F) * E_fd)
            dV_r_dt      = (1 / self.T_A) * (-V_r + (self.K_A * R_F) - (self.K_A * self.K_F / self.T_F) * E_fd + self.K_A * (self.V_ref - V_t))
            dE_fd_dt     = (1 / self.T_E) * (-(self.K_E + 0.098 * np.e**(E_fd*0.55)) * E_fd + V_r)
            
            if self.model_flag=="SM_AVR_GOV":        # Governor equations # recheck it after the meeting
                dP_m_dt  = (1 / self.T_ch) * (-P_m  + P_sv) # 4.110  from dynamics dT_m_dt = - P_m / (2 * H) + P_sv 4.100 + check draw.io
                dP_sv_dt = (1 / self.T_sv) * (-P_sv + self.P_c - (1/self.R_d) * omega)
                return [dtheta_dt, domega_dt, dE_d_dash_dt, dE_q_dash_dt, dR_F_dt, dV_r_dt, dE_fd_dt, dP_m_dt, dP_sv_dt]#4.116 recheck it after the meeting
            return [dtheta_dt, domega_dt, dE_d_dash_dt, dE_q_dash_dt, dR_F_dt, dV_r_dt, dE_fd_dt]
        
    def solve(self, x0, method):
        """
        Solve the differential equations for the synchronous machine model.

        Parameters:
        - x0: list of initial state variables

        Returns:
        - solution: solution of the differential equations
        """

        t_span, t_eval = set_time(self.t, self.num_of_points)
        """
        # Initial state
        if self.model_flag=="SM_IB" or self.model_flag=="SM":
            x0 = [self.theta, self.omega, self.E_d_dash, self.E_q_dash]
        if self.model_flag=="SM_AVR":
            x0 = [self.theta, self.omega, self.E_d_dash, self.E_q_dash, self.R_F, self.V_r, self.E_fd]
        if self.model_flag=="SM_AVR_GOV":
            x0 = [self.theta, self.omega, self.E_d_dash, self.E_q_dash, self.R_F, self.V_r, self.E_fd, self.P_m, self.P_sv]
        """
        if method:
            solution = solve_ivp(self.synchronous_machine_equations, t_span, x0, t_eval=t_eval)
        else:
            x0[1] = x0[1] / self.omega_B
            solution = solve_ivp(self.synchronous_machine_equations_v2, t_span, x0, t_eval=t_eval)
        return solution
    
