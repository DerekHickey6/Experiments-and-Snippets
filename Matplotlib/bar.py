from matplotlib import pyplot as plt
import numpy as np

# Median Developer Salaries by Age
# Ages for x-axis
ages_x = [25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35]

x_indexes = np.arange(len(ages_x))
bar_width = 0.3

# Median salaries for y-axis of ages on x-axis
dev_y = [38496, 42000, 46752, 49320, 53200,
         56000, 62316, 64928, 67317, 68748, 73752]

plt.bar(x_indexes - bar_width, dev_y, width=bar_width, color='#444444', linestyle='--', label="All devs")

# Median Python Developer Salaries by Age
py_dev_y = [45372, 48876, 53850, 57287, 63016,
            65998, 70003, 70000, 71496, 75370, 83640]

plt.bar(x_indexes, py_dev_y, width=bar_width, color='#5a7d9a', linestyle='-.', linewidth=3,  label="Python devs")

# Median JavaScript Developer Salaries by Age
js_dev_y = [37810, 43515, 46823, 49293, 53437,
            56373, 62375, 66674, 68745, 68746, 74583]

plt.bar(x_indexes + bar_width, js_dev_y, width=bar_width, color='#559d5a', linestyle='-', label="JavaScript devs")

plt.title("Median income (USD) by age")
plt.xlabel("Ages")
plt.ylabel("Median Salary (USD)")
plt.xticks(ticks=x_indexes, labels=ages_x)
plt.legend()


plt.tight_layout()
plt.savefig("bar.png")

plt.show()
