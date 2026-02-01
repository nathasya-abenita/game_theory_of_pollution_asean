import numpy as np
import scipy as sp
from scipy.optimize import minimize_scalar # Function to call Brent algorithm
from scipy.optimize import minimize # Function to minimize multivariable scalar function
from scipy.optimize import Bounds
from scipy.special import expit # Sigmoid function

import matplotlib as mpl
import matplotlib.pyplot as plt

mpl.rcParams.update({
    # --- Figure ---
    "figure.figsize": (7.2, 3.6),     # full two-column width
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "savefig.format": "pdf",

    # --- Fonts ---
    "font.family": "serif",
    "font.size": 10,
    "axes.labelsize": 10,
    "legend.fontsize": 9,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,

    # --- Axes ---
    "axes.linewidth": 1.0,
    "axes.spines.top": False,
    "axes.spines.right": False,

    # --- Ticks ---
    "xtick.direction": "in",
    "ytick.direction": "in",
    "xtick.major.size": 4,
    "ytick.major.size": 4,

    # --- Lines ---
    "lines.linewidth": 1.0,

    # --- Legend ---
    "legend.frameon": False,
})

class PollutionGame:

    def __init__ (self, n, beta, psi, neigh, lambd, initial_x_list, mu = None):
        # Save parameters
        self.n = n # number of players
        self.beta = beta
        self.psi = psi
        self.neigh = neigh
        self.lambd = lambd
        self.initial_x_list = initial_x_list

        # Special parameter for cooperative game (diplomatic influence)
        if type(mu) != type(None):
            self.mu = mu
    
    def check_x_list (self, x_list):
        if type(x_list) == type(None):
            return self.initial_x_list
        else:
            return x_list
            
    def cost_func (self, i, u_list, x_list = None):
        """ Cost function for player i """

        # Determine state variable to follow initial or new input
        x_list = self.check_x_list(x_list)

        # Create the cost function
        f =  lambda u : np.exp( self.beta * ((1 - 2 * u) * x_list[i] + \
        np.sum([(self.psi[j][i]*(1 - u_list[j]) * x_list[j]) if j!=i else 0 for j in self.neigh[i]]))) - \
        1 + (self.lambd * u)

        # Return cost
        return f(u_list[i])
    
    def joint_cost_func_lambda (self, x_list):
        return lambda u_list : sum([(self.mu[i] * self.cost_func(i, u_list, x_list)) for i in range (self.n)])
    
    def joint_gradient (self, u_list, x_list):
        grad_list = []
        for i in range (self.n):
            grad_sum = 0
            for j in range (self.n):
                pol_dynamics = (1-2*u_list[j])*x_list[j] + sum([(self.psi[k][j] * (1-u_list[k]) * x_list[k]) if (j!=k) else 0 for k in range (self.n)])
                if i == j:
                    grad = -2*x_list[i]*self.beta*np.exp(self.beta*pol_dynamics) + self.lambd
                else:
                    grad = -self.psi[i][j]*x_list[i]*self.beta*np.exp(self.beta*pol_dynamics)
                grad_sum += self.mu[i]*grad
                grad_list.append(grad_sum)
        return grad_list
    
    def joint_gradient_lambda (self, x_list):
        return lambda u_list : self.joint_gradient(u_list, x_list)
    
    def solve_best_response (self, i, u_list, x_list = None):
        """ Best response for player i given opponents' decisions analytically """

        # Determine state variable to follow initial or new input
        x_list = self.check_x_list(x_list)

        # Apply best response mapping
        if x_list[i] == 0:
            return 0
        else:
            # Compute sum term depending on other players' decisions
            sum = np.sum([(self.psi[j][i]*(1-u_list[j])*x_list[j]) if j!=i else 0 for j in self.neigh[i]])

            # Determine best response analytically
            u = 0.5 - 1/(2*self.beta*x_list[i])*np.log(self.lambd/(2*self.beta*x_list[i])) + 1/(2*x_list[i])*sum

            # Piece-wise condition for best response
            if u < 0:
                return 0
            elif u <= 1:
                return u
            else: # u > 1
                return 1
            
    def validate_nash_equilibrium (self, u_list):
        # Compute best response analytically
        u_analytical = np.zeros(self.n)
        for i in range (self.n):
            u_analytical[i] = self.solve_best_response(i, u_list)

        # Compute error
        error = np.array(u_list) - u_analytical
        error = np.sum(np.abs(error)) 

        # Check whether the values are the same
        if error < 1e-6:
            print('Numerical solution fulfills the definition of best response for each player with error of 1e-6')
            return True
        else:
            print("Numerical solution DOESN'T fulfills the definition of best response for each player with error of 1e-6")
            return False
    
    def solve_static_game_noncoop (self, x_list = None):

        # 0. Determine state variable to follow initial or new input
        x_list = self.check_x_list(x_list)

        # 1. Choose initial point of Nash equilibrium
        u_list = np.array([0.5 for _ in range (self.n)])
        
        # 2. Iterate best response for each player
        iter_max = 100 # number of maximum iteration
        iter = 0
        is_convergent = False
        while (iter <= iter_max) and (is_convergent == False):
            # print("Iteration " + str(iter+1) + ":")
            u_list_old = u_list.copy()

            # Iterating best response for each player
            for i in range (self.n):
                u_list[i] = self.solve_best_response(i, u_list_old) # updating best respons for player i

            # Convergence checking
            err = np.sum(abs(u_list - u_list_old))
            if err < 1e-6:
                is_convergent = True

            # Updating num of iteration
            iter += 1

        # 3. Output (u_list is the nash equilibrium of the game)
        return u_list
    
    def x_dot (self, i, u_list, x_list):
        """ First derivative of state variable for player i """
        return (1 - 2 * u_list[i]) * x_list[i] + \
                sum([(self.psi[j][i] * (1-u_list[j]) * x_list[j]) if (j!=i) else 0 for j in self.neigh[i]])
    
    def update_x (self, u_list, x_list, step_size):
        """ Apply runge-kutta order of four to update state variable for all players """

        # Compute first term
        F1 = np.array([self.x_dot(i, u_list, x_list) for i in range (self.n)])

        # Compute second term
        x1 = x_list + step_size / 2 * F1
        F2 = np.array([self.x_dot(i, u_list, x1) for i in range (self.n)])

        # Compute third term
        x2 = x_list + step_size / 2 * F2 
        F3 = np.array([self.x_dot(i, u_list, x2) for i in range (self.n)])

        # Compute fourth term
        x3 = x_list + step_size * F3
        F4 = np.array([self.x_dot(i, u_list, x3) for i in range (self.n)])

        # Finalize new state variables
        return x_list + step_size / 6 * (F1 + 2*F2 + 2*F3 + F4)
        
    
    def solve_diff_game_noncoop (self, duration, step_size = 0.01):
        # Compute number of iterations
        n_iter = int(duration / step_size) + 1

        # Initialize time axis
        t_list = [(t * step_size) for t in range (n_iter + 1)]

        # Initialize arrays to save state and decision variables throughout the time
        x_list_final = np.empty((self.n, n_iter + 1))
        u_list_final = np.empty((self.n, n_iter + 1))

        # Compute initial decision
        u_list_final[:, 0] = self.solve_static_game_noncoop()

        # Save initial condition for state variable
        x_list_final[:, 0]  = self.initial_x_list

        # Loop over each time
        for t in range (n_iter):
            # Update state variable
            x_list_final[:, t + 1] = self.update_x(u_list_final[:, t], x_list_final[:, t], step_size)

            # Update decision variable
            u_list_final[:, t + 1] = self.solve_static_game_noncoop(x_list_final[:, t])
        return t_list, u_list_final, x_list_final
    
    def sigm(self, i, j, u_list, x_list):
        # Compute cost function for player i
        c_i = self.cost_func(i, u_list, x_list)

        # Compute cost function for player j
        c_j = self.cost_func(j, u_list, x_list)

        # Return the cost difference in sigmoid function
        return expit(c_i - c_j) # 1/(1 + np.exp(c_j-c_i))
    
    def u_dot (self, i, u_list, x_list, typ):
        """ First derivative of control variable for player i """

        if typ.lower() == "basic":
            return sum([(u_list[j] - u_list[i]) for j in self.neigh[i]]) / self.n
        elif typ.lower() == 'advanced':
            return sum([self.sigm(i, j, u_list, x_list) * (u_list[j] - u_list[i]) for j in self.neigh[i]]) / self.n
        else:
            raise ValueError ('Accepted imitation types: ["advanced", "basic"]')
    
    def update_u (self, u_list, x_list, step_size, typ):
        """ 
        Apply runge-kutta order of four to update control variable for all players
        following imitation game with chosen type (basic or advanced)
        """
        
        # Compute first term
        F1 = np.array([self.u_dot(i, u_list, x_list, typ) for i in range (self.n)])

        # Compute second term
        u1 = u_list + step_size / 2 * F1
        F2 = np.array([self.u_dot(i, u1, x_list, typ) for i in range (self.n)])

        # Compute third term
        u2 = u_list + step_size / 2 * F2 
        F3 = np.array([self.u_dot(i, u2, x_list, typ) for i in range (self.n)])

        # Compute fourth term
        u3 = u_list + step_size * F3
        F4 = np.array([self.u_dot(i, u3, x_list, typ) for i in range (self.n)])

        # Finalize new state variables
        return u_list + step_size / 6 * (F1 + 2*F2 + 2*F3 + F4)
        
    
    def solve_diff_game_imitate (self, duration, step_size = 0.01, typ = 'advanced'):
        """ Solve imitation game with chosen type (basic or advanced) """

        # Compute number of iterations
        n_iter = int(duration / step_size) + 1  # total iteration
        initial_iter = int(0.01 * n_iter)        # iteration for initial noncoop. game

        # Initialize time axis
        t_list = [(t * step_size) for t in range (n_iter + 1)]

        # Initialize arrays to save state and decision variables throughout the time
        x_list_final = np.empty((self.n, n_iter + 1))
        u_list_final = np.empty((self.n, n_iter + 1))

        # Compute initial decision
        initial_duration = initial_iter * step_size
        _, u_list_final[:, :initial_iter+2], x_list_final[:, :initial_iter+2] = self.solve_diff_game_noncoop(initial_duration, step_size)
         
        # Loop over each time
        for t in range (initial_iter+1, n_iter):
            # Update state variable
            x_list_final[:, t + 1] = self.update_x(u_list_final[:, t], x_list_final[:, t], step_size)

            # Update decision variable
            u_list_final[:, t + 1] = self.update_u(u_list_final[:, t], x_list_final[:, t], step_size, typ)
        return t_list, u_list_final, x_list_final
    
    def solve_diff_game_coop (self, duration, step_size = 0.01):
        # Define minimization parameters
        bounds = Bounds([0 for _ in range (self.n)], [1 for _ in range (self.n)])
        initial_guess = np.array([0.5 for _ in range (self.n)])
        
        # Compute number of iterations
        n_iter = int(duration / step_size) + 1

        # Initialize time axis
        t_list = [(t * step_size) for t in range (n_iter + 1)]

        # Initialize arrays to save state and decision variables throughout the time
        x_list_final = np.empty((self.n, n_iter + 1))
        u_list_final = np.empty((self.n, n_iter + 1))
        cost_final = np.empty(n_iter + 1)

        # Compute initial decision
        x_list = self.initial_x_list
        res = minimize(self.joint_cost_func_lambda(x_list), initial_guess, bounds=bounds,
              options={'gtol': 1e-10, 'maxiter': 8000})
        u_list_final[:, 0] = res.x
        cost_final[0] = res.fun
        self.print_when_fail(res.success)

        # Save initial condition for state variable
        x_list_final[:, 0]  = self.initial_x_list

        # Loop over each time
        for t in range (n_iter):
            # Update state variable
            x_list_final[:, t + 1] = self.update_x(u_list_final[:, t], x_list_final[:, t], step_size)

            # Update decision variable
            x_list = x_list_final[:, t]
            res = minimize(self.joint_cost_func_lambda(x_list), initial_guess, bounds=bounds,
                options={'gtol': 1e-10, 'maxiter': 8000})
            u_list_final[:, t + 1] = res.x
            cost_final[t + 1] = res.fun
            self.print_when_fail(res.success)
        return t_list, u_list_final, x_list_final, cost_final
    
    def print_when_fail (self, status):
        if not status:
            print('Minimization with L-BFGS-B failed!')

    def display_diff_game (self, t_list, u_list_final, x_list_final, labels, filename = 'game.png'):
        # Create file name with suitable path
        filename = f'./output/figs/{filename}'

        # Choice of colors and linestyles
        colors = plt.cm.tab10.colors
        linestyles = ['-' for _ in range (self.n)] #["-", "--", "-.", ":", "-", "--", "-.", ":", "-", "--"]

        # Initialize plot
        plt.clf()
        fig, axs = plt.subplots(2, 1, sharex=True)

        # State variables
        for i in range(self.n):
            axs[0].plot(t_list, x_list_final[i, :], color=colors[i], linestyle=linestyles[i], label=labels[i])
        axs[0].set_ylabel("$x_i(t)$")
        axs[0].grid(True, linestyle=":", linewidth=0.4, alpha=0.4)
        # axs[0].legend(bbox_to_anchor=(1.02, 1), loc="upper left")

        # Decision variables
        for i in range(self.n):
            axs[1].plot(t_list, u_list_final[i, :], color=colors[i], linestyle=linestyles[i])
        axs[1].set_ylim(0, 1)
        axs[1].set_xlim(0, t_list[-1])
        axs[1].grid(True, linestyle=":", linewidth=0.4, alpha=0.4)
        axs[1].set_ylabel("$u_i(t)$")
        axs[1].set_xlabel("$t$")

        # Save plot
        plt.tight_layout()
        plt.savefig(filename)
    
    def display_joint_cost(self, t_list, cost_list, filename):
        # Create file name with suitable path
        filename = f'./output/figs/{filename}'

        # Plot
        plt.clf()
        plt.plot(t_list, cost_list)
        plt.xlabel("$t$"); plt.ylabel("$c(t)$")
        plt.grid(True, linestyle=":", linewidth=0.4, alpha=0.4)

        # Save plot
        plt.tight_layout()
        plt.savefig(filename)

