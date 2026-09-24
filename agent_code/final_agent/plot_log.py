from pathlib import Path
import matplotlib.pyplot as plt

log_path = Path("agent_code/final_agent/logs/final_agent.log")
rows = []

with log_path.open() as log:
    for line in log:
        if "[ROLLING_STATS] " not in line:
            continue

        payload = line.split("[ROLLING_STATS] ", 1)[1]
        fields = dict(item.split("=", 1) for item in payload.split())
        rows.append({key: float(value) for key, value in fields.items()})

if not rows:
    raise ValueError("No ROLLING_STATS entries found.")

metrics = [
    ("coins", "Average coins"),
    ("kills", "Average kills"),
    ("crates", "Average crates"),
    ("suicide_ratio", "Suicide ratio"),
    ("survived_steps", "Average steps in survived rounds"),
]

steps = [row["steps"] for row in rows]
fig, axes = plt.subplots(5, 1, figsize=(10, 12), sharex=True)

for ax, (key, label) in zip(axes, metrics):
    ax.plot(steps, [row[key] for row in rows])
    ax.set_ylabel(label)
    ax.grid(alpha=0.3)

axes[3].set_ylim(0, 1)
axes[-1].set_xlabel("Training steps since process start")
fig.suptitle("Rolling averages over the last 10 rounds")
fig.tight_layout()
fig.savefig("final_agent_learning_curves.png", dpi=200)
plt.show()