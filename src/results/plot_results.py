import matplotlib.pyplot as plt

with open("rewards_DQN2.txt", "r") as file:
    lines = file.readlines()

data = [list(map(float, line.split())) for line in lines]

X = [row[0] for row in data]
Y1 = [row[1] for row in data]
Y2 = [row[2] for row in data]
Y3 = [row[3] for row in data]

plt.figure(figsize=(10, 8))

plt.subplot(3, 1, 2)
plt.plot(Y1, label="Reward")
plt.xlabel("Episode")
plt.ylabel("Reward")
plt.legend()

plt.subplot(3, 1, 1)
plt.plot(Y2, label="Timestep")
plt.xlabel("Episode")
plt.ylabel("Timestep")
plt.legend()

plt.subplot(3, 1, 3)
plt.plot(Y3, label="Score")
plt.xlabel("Episode")
plt.ylabel("Score")
plt.legend()

plt.tight_layout()
plt.show()
