import torch
import numpy as np


path = "/home/eric/Desktop/bomberman_rl/agent_code/kill_agent/kill_agent_saved_model.pt"

state = torch.load(path, map_location="cpu", weights_only = False)
epsilon_min = 0.05
epsilon_max = 1.0
decay_const = 569840.0

print("Please input desired epsilon done : ")
epsilon = float(input())

arg = (epsilon - epsilon_min) / (epsilon_max - epsilon_min)
#steps = - decay_const * np.log(arg)
state["steps_done"] = - decay_const * np.log(arg)
torch.save(state,path)
#print(epsilon_min + (epsilon_max - epsilon_min) * np.exp(- steps/ dcay_const))
print("New reset. New espilon = " + str(epsilon))
