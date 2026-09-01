import torch

path = "/home/eric/Desktop/bomberman_rl/agent_code/kill_agent/kill_agent_saved_model.pt"

state = torch.load(path, map_location="cpu", weights_only = False)
print("Please input steps done : ")
state["steps_done"] = int(input())
torch.save(state,path)

print("epsilon reset: steps done = " + str(state["steps_done"]))