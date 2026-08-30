import numpy as np

eps_min = 0.05
eps_max = 1.0
def get_eps(n_games, tau):

    steps = n_games * 400
    return eps_min + (eps_max - eps_min) * np.exp(-1.0 * steps / tau)

def get_tau(n_games, eps_end):
    steps = n_games * 400
    val = (eps_max - eps_min) / (eps_end - eps_min)

    return round(steps/np.log(val))

tau = get_tau(n_games=5500, eps_end=0.07)
print(tau)