from matplotlib import pyplot as plt
import numpy as np

x = np.arange(100)

fig, axs = plt.subplots(2, 2)

axs[0, 0].plot(x, np.sin(x))
axs[0, 0].set_title("Sine Wave")

axs[0, 1].plot(x, np.cos(x))
axs[0, 1].set_title("Cosine Wave")

axs[1, 0].plot(x, np.random.random(100))
axs[1, 0].set_title("Random function")

axs[1, 1].plot(x, np.log(x))
axs[1, 1].set_title("Log Function")

fig.suptitle('Subplotting Example')

# x1, y1 = np.random.random(100), np.random.random(100)
# x2, y2 = np.arange(100), np.random.random(100)

# # plt.figure(1)
# # plt.scatter(x1, y1)

# # plt.figure(2)
# # plt.plot(x2, y2)

plt.tight_layout()

plt.savefig('subplot.png', dpi=300, transparent=False, bbox_inches="tight", pad_inches=0.2)

plt.show()