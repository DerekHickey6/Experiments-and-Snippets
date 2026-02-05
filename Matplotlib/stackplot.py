from matplotlib import pyplot as plt
# StackPlot: Good for track a total and a breakdown by category
# Shows scores stacked on top of eache other
plt.style.use("fivethirtyeight")

minutes = [1, 2, 3, 4, 5, 6, 7, 8, 9]

# Cumulative score
player1 = [1, 2, 3, 3, 4, 4, 4, 4, 5]
player2 = [1, 1, 1, 1, 2, 2, 2, 3, 4]
player3 = [1, 1, 1, 2, 2, 2, 3, 3, 3]

labels = ['player1', 'player2', 'player3']
plt.stackplot(minutes, player1, player2, player3, labels=labels)

plt.legend()
plt.title("My Stack Plot")
plt.tight_layout()
plt.savefig("stackplot.png")
plt.show()