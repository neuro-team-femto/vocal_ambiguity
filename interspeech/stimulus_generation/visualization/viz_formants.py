import json
import seaborn as sns
import matplotlib.pyplot as plt

# Load data from JSON file
filename = 'formant_results.json'
with open(filename, 'r') as file:
    formants = json.load(file)

sns.set() # Use seaborn's default style to make attractive graphs
plt.rcParams['figure.dpi'] = 400 # Show nicely large images in this notebook

fig, ax_kwargs = plt.subplots()

ax_kwargs.axvline(c='grey', lw=1)
ax_kwargs.axhline(c='grey', lw=1)

colours = ['g', 'r', 'y', 'k', 'b', 'm', 'w']

for word in formants:
    x = formants[word]['f1']
    y = formants[word]['f2']
    ax_kwargs.scatter(x, y, s=500, marker=f"${word}$")
ax_kwargs.set_title(f'F1 F2 plot')

plt.ylim(bottom=4)
plt.xlim(left=0)

ax_kwargs.set_xlabel("F1 Hz")
ax_kwargs.set_ylabel("F2 Hz")

fig.subplots_adjust(hspace=0.25)
fig1 = plt.gcf()
plt.show()