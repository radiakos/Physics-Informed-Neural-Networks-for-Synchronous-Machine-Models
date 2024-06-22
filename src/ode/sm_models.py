import numpy as np
from scipy.integrate import solve_ivp
from src.functions import set_time

class SynchronousMachineModels():
    def __init__(self, t, num_of_points, x, params, model_flag):
        """                     
        Define the differential equations for the synchronous machine with Automatic Voltage Regulator (AVR).

        Parameters:
            t: float
                The current time.
            num_of_points: int
                The number of points to evaluate the solution.
            x: array-like, shape (2 or 4 or 7 or 9,)
                The current values of theta, omega, E_d_dash, and E_q_dash.
            params: array-like, shape (7 or 13 or 19 or 22(P_m not parameter),)
                The parameters of the synchronous machine.
            model_flag
        """
        self.t = t
        self.num_of_points = num_of_points
        self.model_flag = model_flag
        self.x = x
        self.params = params
        # parameters of the synchronous machine
        self.X_d_dash = params[0]
        self.X_q_dash = params[1]
        self.Vs       = params[2]
        self.theta_vs = params[3]
        self.omega_B  = params[4]
        self.H        = params[5]
        
        if not (model_flag=="SM_AVR_GOV"): # if governor exists, then P_m is a control input
            self.P_m  = params[6]
        self.D        = params[7]

        if not (model_flag=="SM_IB"): # not infinite bus parameters
            self.T_q_dash = params[8]
            self.X_q      = params[9]
            self.T_d_dash = params[10]
            self.X_d      = params[11]
            self.R_s      = params[12]
        if not (model_flag=="SM_AVR" or model_flag=="SM_AVR_GOV"): # automatic voltage regulator parameters
            self.E_fd     = params[13]
        if model_flag=="SM_IB":
            self.model= "Infinite bus model"
            #print("Infinite bus model")

        if model_flag=="SM": # 2 axis Synchronous Machine Model
            self.model= "2 axis Synchronous Machine Model"
            #print("2 axis Synchronous Machine Model")

        if model_flag=="SM_AVR":
            self.model= "2 axis Synchronous Machine Model with AVR"
            #print("2 axis Synchronous Machine Model with AVR")
        if model_flag=="SM_AVR" or model_flag=="SM_AVR_GOV": # automatic voltage regulator parameters
            self.K_A      = params[14]
            self.T_A      = params[15]   
            self.K_E      = params[16]
            self.T_E      = params[17]
            self.K_F      = params[18]
            self.T_F      = params[19]
            self.V_ref    = params[20]
          
        
        if model_flag=="SM_AVR_GOV": # governor parameters
            self.model= "2 axis Synchronous Machine Model with AVR and governor"
            #print("2 axis Synchronous Machine Model with AVR and governor")
            self.P_c      = params[21]
            self.R_d      = params[22]
            self.T_ch     = params[23]
            self.T_sv     = params[24]
        

        # initial conditions
        self.theta      = x[0]
        self.omega      = x[1]
        self.E_d_dash   = x[2]
        self.E_q_dash   = x[3]
        if model_flag=="SM_AVR" or model_flag=="SM_AVR_GOV":
            self.R_F    = x[4]
            self.V_r    = x[5]
            self.E_fd   = x[6]

        if model_flag=="SM_AVR_GOV":
            self.P_m    = x[7]
            self.P_sv   = x[8]
            



    def calculate_currents(self, theta, E_d_dash, E_q_dash):
        Rs=0.0
        Re=0.0
        Xep=0.0
        alpha = [[(Rs+Re), -(self.X_q_dash+Xep)], [(self.X_d_dash+Xep), (Rs+Re)]]
        beta = [[E_d_dash - self.Vs*np.sin(theta-self.theta_vs)], [E_q_dash - self.Vs*np.cos(theta-self.theta_vs)]]
        
        inv_alpha = np.linalg.inv(alpha)
        # Calculate I_d and I_q
        I_t = np.matmul(inv_alpha, beta)
        I_d = I_t[0][0]
        I_q = I_t[1][0]
        return I_d, I_q

    def calculate_voltages(self, theta, I_d, I_q):
        Re=0.0
        Xep=0.0
        V_d = Re*I_d - Xep*I_q + self.Vs*np.sin(theta-self.theta_vs)
        V_q = Re*I_q + Xep*I_d + self.Vs*np.cos(theta-self.theta_vs)
        V_t = np.sqrt(V_d**2 + V_q**2)# equal to Vs
        return V_t

    def synchronous_machine_equations(self, t, x):
        
        if self.model_flag=="SM_IB" or self.model_flag=="SM":
            theta, omega, E_d_dash, E_q_dash = x
        if self.model_flag=="SM_AVR":
            theta, omega, E_d_dash, E_q_dash, R_F, V_r, E_fd = x
        if self.model_flag=="SM_AVR_GOV":
            theta, omega, E_d_dash, E_q_dash, R_F, V_r, E_fd, P_m, P_sv = x

        # Calculate currents from algebraic equations
        I_d, I_q = self.calculate_currents(theta, E_d_dash, E_q_dash)

        if self.model_flag=="SM_AVR" or self.model_flag=="SM_AVR_GOV": # calculate V_t from algebraic equations
            V_t = self.calculate_voltages(theta, I_d, I_q)
        
        # Calculate theta derivative
        dtheta_dt = omega
        
        # Calculate omega derivative
        if self.model_flag=="SM_AVR_GOV": # calculate omega derivative from algebraic equations
            domega_dt = (self.omega_B / (2 * self.H)) * (P_m - E_d_dash * I_d - E_q_dash * I_q - (self.X_q_dash - self.X_d_dash) * I_q * I_d - self.D * omega ) # I think multiply by omega_B is needed
        else:
            domega_dt = (self.omega_B / (2 * self.H)) * (self.P_m - E_d_dash * I_d - E_q_dash * I_q - (self.X_q_dash - self.X_d_dash) * I_q * I_d - self.D * omega )
        
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
        if self.model_flag=="SM_AVR" or self.model_flag=="SM_AVR_GOV":
            dE_q_dash_dt = (1 / self.T_d_dash) * (- E_q_dash - I_d * (self.X_d - self.X_d_dash) + E_fd)
            dE_d_dash_dt = (1 / self.T_q_dash) * (- E_d_dash + I_q * (self.X_q - self.X_q_dash))
            dR_F_dt      = (1 / self.T_F) * (-R_F + (self.K_F / self.T_F) * E_fd)
            dV_r_dt      = (1 / self.T_A) * (-V_r + (self.K_A * R_F) - (self.K_A * self.K_F / self.T_F) * E_fd + self.K_A * (self.V_ref - V_t))
            dE_fd_dt     = (1 / self.T_E) * (-(self.K_E + 0.098 * np.e**(E_fd*0.55)) * E_fd + V_r)
            
            if self.model_flag=="SM_AVR_GOV":        # Governor equations # recheck it after the meeting
                dP_m_dt  = (1 / self.T_ch) * (-P_m  + P_sv) # 4.110  from dynamics dT_m_dt = - P_m / (2 * H) + P_sv 4.100 + check draw.io
                dP_sv_dt = (1 / self.T_sv) * (-P_sv + self.P_c - (1/self.R_d) * (omega/self.omega_B)) 
                return [dtheta_dt, domega_dt, dE_d_dash_dt, dE_q_dash_dt, dR_F_dt, dV_r_dt, dE_fd_dt, dP_m_dt, dP_sv_dt]#4.116 recheck it after the meeting
            return [dtheta_dt, domega_dt, dE_d_dash_dt, dE_q_dash_dt, dR_F_dt, dV_r_dt, dE_fd_dt]
        
    def synchronous_machine_equations2(self, t, x):
        
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
        dtheta_dt = omega* self.omega_B
        
        # Calculate omega derivative
        if self.model_flag=="SM_AVR_GOV": # calculate omega derivative from algebraic equations
            domega_dt = (1 / (2 * self.H)) * (P_m - E_d_dash * I_d - E_q_dash * I_q - (self.X_q_dash - self.X_d_dash) * I_q * I_d - self.D * omega * self.omega_B ) # I think multiply by omega_B is needed
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
        t_span, t_eval = set_time(self.t,self.num_of_points)
        # Initial state
        """
        if self.not_ib_flag or self.ib_flag:
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
            solution = solve_ivp(self.synchronous_machine_equations2, t_span, x0, t_eval=t_eval)
        return solution, self.model
    
