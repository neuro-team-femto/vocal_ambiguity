import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import json

# Load data from JSON file
filename = 'english_results_pill.json'
with open(filename, 'r') as file:
    data = json.load(file)

# Get word
word = filename.split(".")[0].split("_")[-1]

# Extract data from JSON
f1 = data[word]['f1']
f2 = data[word]['f2']
log_prob_diff = data[word]['log_prob_diff']

# Create 3D surface plot
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

ax.plot_trisurf(f1, f2, log_prob_diff, cmap='viridis')

# Set labels and title
ax.set_xlabel('f1')
ax.set_ylabel('f2')
ax.set_zlabel('log_prob_diff')
ax.set_title(word)

# Show the plot
plt.show()
