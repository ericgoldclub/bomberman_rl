import pickle

with open("q_table.pkl", "rb") as file:
    q_table = pickle.load(file)

print("Anzahl gespeicherter Q-Werte:", len(q_table))

for i, ((state, action), value) in enumerate(q_table.items()):
    print(state, action, value)

    if i >= 20:
        break
