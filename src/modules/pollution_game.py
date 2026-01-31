import numpy as np
import scipy as sp
from scipy.optimize import minimize_scalar # Function to call Brent algorithm
import matplotlib.pyplot as plt
from scipy.special import expit # Sigmoid function

class PollutionGame:

    def __init__ (self, n, beta, psi, neigh, lambd, initial_x_list):
        # Save parameters
        self.n = n # number of players
        self.beta = beta
        self.psi = psi
        self.neigh = neigh
        self.lambd = lambd
        self.x_list = initial_x_list
            
    def cost_func (self, i, u_list):
        """ Cost function for player i """

        return lambda u : np.exp( self.beta * ((1 - 2 * u) * self.x_list[i] + \
        np.sum([(self.psi[j][i]*(1 - u_list[j]) * self.x_list[j]) if j!=i else 0 for j in self.neigh[i]]))) - \
        1 + (self.lambd * u)
    
    def solve_best_response (self, i, u_list):
        """ Best response for player i given opponents' decisions analytically """

        if self.x_list[i] == 0:
            return 0
        else:
            # Compute sum term depending on other players' decisions
            sum = np.sum([(self.psi[j][i]*(1-u_list[j])*self.x_list[j]) if j!=i else 0 for j in self.neigh[i]])

            # Determine best response analytically
            u = 0.5 - 1/(2*self.beta*self.x_list[i])*np.log(self.lambd/(2*self.beta*self.x_list[i])) + 1/(2*self.x_list[i])*sum

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
        error = np.sum(error**2) ** 0.5

        # Check whether the values are the same
        if error < 1e-6:
            print('Numerical solution fulfills the definition of best response for each player with error of 1e-6')
            return True
        else:
            print("Numerical solution DOESN'T fulfills the definition of best response for each player with error of 1e-6")
            return False

    def solve_static_game_numerical (self):
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
                opt = minimize_scalar(self.cost_func(i, u_list_old), bounds=(0, 1), method='bounded')
                u_list[i] = opt.x # updating best respons for player i

            # Convergence checking
            err = 0
            for i in range (self.n):
                err += abs(u_list[i] - u_list_old[i])
            if err < 1e-6:
                is_convergent = True

            # Updating num of iteration
            iter += 1

        # 3. Output (u_list is the nash equilibrium of the game)
        return u_list
