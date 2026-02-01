import numpy as np
from modules.pollution_game import PollutionGame

if __name__ == "__main__" :
    # Read psi
    psi_list = np.load("./output/vars/psi_list.npy")
    
    # Defining parameters and functions
    n = 10 # number of players
    T, step_size = 12*3, 0.01
    beta, lambd = 10, 4
    neigh = [[i for i in range (n)] for _ in range (n)]
    mu = [1 for _ in range (n)]
    psi = psi_list[0]

    # Define labels for players
    labels = ["Indonesia", "Vietnam", "Thailand", "Malaysia", "The Philippines",
              "Singapore", "Myanmar", "Cambodia", "Laos", "Brunei"]

    # Initialize x (pollution stock) for each countries
    initial_x_list = np.array([625, 311, 289, 249, 154, 53, 49, 16, 7, 7])
    initial_x_list = initial_x_list / 100

    # Call game class
    game = PollutionGame(n, beta, psi, neigh, lambd, initial_x_list, mu=mu)

    # Simulate noncooperative game, then plot
    t_list, u_list_final, x_list_final, cost_list = game.solve_diff_game_coop(duration=T, step_size=step_size)
    game.display_diff_game(t_list, u_list_final, x_list_final, labels, 'example_coop.png')
    game.display_joint_cost(t_list, cost_list, 'example_coop_joint_cost.png')