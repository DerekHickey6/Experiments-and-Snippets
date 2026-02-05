from matplotlib import pyplot as plt

plt.style.use("fivethirtyeight")

# Language Popularity
slices = [59219, 55466, 47544, 36443, 35917]
labels = ['JavaScript', 'HTML/CSS', 'SQL', 'Python', 'Java']
explode = [0, 0, 0, 0.1, 0]
colors = [
    "#FFE0EB",
    "#FFD1DC",
    "#FFC2D1",
    "#FFB3C6",
    "#FF8FAB"
]

plt.pie(slices, explode=explode, shadow=True, startangle = 90, autopct='%1.1f%%',colors=colors, labels=labels, wedgeprops={'edgecolor': 'black'})

plt.title("5 Most Common Programming Languages")
plt.savefig("piechart.png")
plt.tight_layout()
plt.show()