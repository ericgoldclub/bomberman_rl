import os

import numpy as np
import torch

path = os.path.join(os.path.dirname(__file__), "Ultra_Network_agent_saved_model_v2.pt")

if not os.path.isfile(path):
    raise FileNotFoundError(f"No Ultra Network checkpoint found at {path}")

state = torch.load(path, map_location="cpu", weights_only = False)
epsilon_min = 0.05
epsilon_max = 1.0
decay_const = 569840.0

print("Please input desired epsilon done : ")
epsilon = float(input())
if not epsilon_min < epsilon <= epsilon_max:
    raise ValueError(f"epsilon must be in ({epsilon_min}, {epsilon_max}]")

arg = (epsilon - epsilon_min) / (epsilon_max - epsilon_min)
#steps = - decay_const * np.log(arg)
state["steps_done"] = int(round(-decay_const * np.log(arg)))
torch.save(state,path)
#print(epsilon_min + (epsilon_max - epsilon_min) * np.exp(- steps/ dcay_const))
print("New reset. New espilon = " + str(epsilon))
