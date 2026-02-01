import numpy as np
from modules.pollution_game import PollutionGame

if __name__ == "__main__" :
    
    # Defining parameters and functions
    n = 10 # number of players
    T, step_size = 12*3, 0.01
    beta, lambd = 10, 4
    psi = [[0.1 for _ in range (n)] for _ in range (n)]
    neigh = [[1,2,3,4], [1,2,3],
            [1,2], [1,2],
            [1,2,5,6], [1,2],
            [1,2], [1,2],
            [1,2,8,9], [1,2,7,8,9]]
    x_list_final, u_list_final = [[] for _ in range (n)], [[] for _ in range (n)]

    # Define labels for players
    labels = ["Indonesia", "Vietnam", "Thailand", "Malaysia", "The Philippines",
              "Singapore", "Myanmar", "Cambodia", "Laos", "Brunei"]

    # Initialize x (pollution stock) for each countries
    initial_x_list = np.array([
        0.5, 0.5, 0.5, 0.5, 0.5,
        0.2, 0.2, 0.4, 0.4, 0.7
    ])

    # Call game class
    game = PollutionGame(n, beta, psi, neigh, lambd, initial_x_list)

    # Simulate noncooperative game, then plot
    t_list, u_list_final, x_list_final = game.solve_diff_game_noncoop(duration=T, step_size=step_size)
    game.display_diff_game(t_list, u_list_final, x_list_final, labels, 'example_noncoop.png')

    # Simulate basic imitation game, then plot
    t_list, u_list_final, x_list_final = game.solve_diff_game_imitate(duration=T, step_size=step_size, typ='basic')
    game.display_diff_game(t_list, u_list_final, x_list_final, labels, 'example_basic_imitation.png')

    # Simulate advanced imitation game, then plot
    t_list, u_list_final, x_list_final = game.solve_diff_game_imitate(duration=T, step_size=step_size, typ='advanced')
    game.display_diff_game(t_list, u_list_final, x_list_final, labels, 'example_advanced_imitation.png')