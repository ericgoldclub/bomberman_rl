import numpy as np

eps_min = 0.05
eps_max = 1.0
avg_game_time = float(input("Enter the average game time in seconds: "))
steps_per_game = 400

def get_eps(n_games, tau):

    steps = n_games * 400
    return eps_min + (eps_max - eps_min) * np.exp(-1.0 * steps / tau)

def get_tau(hours, eps_end):
    n_games = hours * 60 * 60 / (avg_game_time * steps_per_game)
    steps = n_games * 400
    val = (eps_max - eps_min) / (eps_end - eps_min)

    return round(steps/np.log(val))

tau = get_tau(hours=int(input("Enter the number of training hours: ")), eps_end=0.07)
print(tau)