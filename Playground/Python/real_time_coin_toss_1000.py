import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

def coin_flip():
    return np.random.randint(0, 2)

heads = 0
tails = 0

plt.ion()
fig, ax = plt.subplots()


for i in range(100):
    if coin_flip() == 0:
        heads += 1
    else:
        tails += 1
    ax.bar(['Heads', 'Tails'], [heads, tails], color=['blue', 'orange'])
    ax.set_xlabel('Coin Side')
    ax.set_ylabel('Count')
    ax.set_title('Coin Flip Simulation')
    ax.set_ylim(0, 100)  # fix y-axis scale
    plt.pause(0.1)

plt.xlabel('Heads: ' + str(heads) + ' Tails: ' + str(tails))
plt.ioff()
plt.show()
