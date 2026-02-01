from modules.pollution_game import PollutionGame
import matplotlib.pyplot as plt
import numpy as np

if __name__ == '__main__':
    # Defining game parameters
    n = 2               # number of players
    beta, lambd = 1, 1
    psi = [[0.1 for _ in range (n)] for _ in range (n)]
    neigh = [[i for i in range (n)] for _ in range (n)]
    initial_x_list = [1, 1]     # initial pollution stock

    # Call game class
    game = PollutionGame(n, beta, psi, neigh, lambd, initial_x_list)

    # Define domain of control variable
    domain = np.arange(0, 1.1, 0.1)

    # Compute mapping of best response function
    map_1 = [game.solve_best_response(0, [None, u]) for u in domain]
    map_2 = [game.solve_best_response(1, [u, None]) for u in domain]
    
    # Plotting BRF
    plt.plot(domain, map_1, color = 'r', label = "$R_1(s_2)$")
    plt.plot(map_2, domain, color = 'b', label = "$R_2(s_1)$")
    #plt.axvline(x = 0.853, color = 'darkgrey', linestyle = '--')
    plt.title("Best Response Function (BRF) for Two Players Game")
    plt.xlabel("$s_2$",fontsize=12) ; plt.ylabel("$s_1$",fontsize=12)
    plt.grid() ; plt.legend(fontsize=12)
    plt.savefig('./output/figs/brf_two_players.png')

    # Solve analytically
    u_analytical = game.solve_static_game_noncoop()
    print('Numerical solution:', u_analytical)

    # Validate numerical solution
    game.validate_nash_equilibrium(u_analytical)