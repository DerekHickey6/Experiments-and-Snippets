from matplotlib import pyplot as plt
import numpy as np
import csv
from collections import Counter
import pandas as pd

plt.style.use('fivethirtyeight')

# Read CSV with pandas
data = pd.read_csv('data.csv')
ids = data['Responder_id']
lang_responses = data['LanguagesWorkedWith']

# Create Counter
language_counter = Counter()

# Update Counter
for response in lang_responses:
    language_counter.update(response.split(";"))

# Separate the languages and counts to plot
languages = []
popularity = []

for item in language_counter.most_common():
    languages.append(item[0])
    popularity.append(item[1])


# Reverse order, most popular at top
languages.reverse()
popularity.reverse()

plt.barh(languages, popularity)

plt.title("Most popular languages")
plt.xlabel("Number of users")
plt.yticks(fontsize=10)

plt.legend()

plt.tight_layout()
plt.savefig("hbar.png")

plt.show()
