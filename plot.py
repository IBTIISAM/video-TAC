import matplotlib.pyplot as plt

# Data
algorithms = ['CLIP(k-means)', 'X-CLIP(k-means)', 'TAC', 'Video-TAC']
accuracy = [58.2 ,59.2, 68.7, 70.7]  # Replace these with your actual clustering accuracy values

# Colors and markers
colors = ['red', 'blue', 'green', 'purple']
markers = ['*', 's', 'o', '^']  # Corresponding to the marker types in the plot

# Create the plot
fig, ax = plt.subplots(figsize=(8, 5))

# Plot each algorithm
for i, (algorithm, acc) in enumerate(zip(algorithms, accuracy)):
    ax.scatter(i + 1, acc, color=colors[i], marker=markers[i], s=100, label=algorithm)
    #ax.text(acc + 1, i + 1, algorithm, fontsize=10, color=colors[i], va="center")  # Add text next to the point

# Add labels and title
ax.set_ylabel('Clustering Accuracy (%) on UCF-101', fontsize=12)
ax.set_xlabel('', fontsize=12)
ax.set_xticks(range(1, len(algorithms) + 1))
ax.set_xticklabels(range(1, len(algorithms) + 1))  # Display only numbers on x-axis
ax.set_title('Externally Guided Clustering', fontsize=14)

# Add grid lines for better readability
ax.grid(axis='y', linestyle='--', alpha=0.7)

# Add the legend
ax.legend(loc='lower right', fontsize=10, title="")

# Adjust layout to avoid cutting off labels
plt.tight_layout()

# Save the plot as an image file
plt.savefig('clustering_accuracy_plot_legend.png', dpi=300, bbox_inches='tight')

# Show the plot
plt.show()
import matplotlib.pyplot as plt



# Data
algorithms = ["CLIP (k-means)", "X-CLIP (k-means)", "TAC", "VIDEO-TAC"]
accuracy = [58.2 ,59.2, 68.7, 70.7]
colors = ["lightcoral", "lightsalmon", "lightblue", "lightgreen"]  # Light colors for bars

# Create the figure and axis
fig, ax = plt.subplots(figsize=(8, 6))

# Plot bars with thinner width
bar_width = 0.5
bars = ax.bar(algorithms, accuracy, color=colors, edgecolor="black", width=bar_width)

# Add labels above the bars
for bar, acc in zip(bars, accuracy):
    ax.text(bar.get_x() + bar.get_width() / 2, acc + 1, f"{acc}%", 
            ha="center", va="bottom", fontsize=10, color="black")

# Customize the plot
ax.set_ylabel("Clustering Accuracy (%)", fontsize=12)
ax.set_xlabel("Algorithms", fontsize=12)
ax.set_title("Clustering Accuracy on UCF-101", fontsize=14, weight="bold")
ax.set_ylim(0, 90)  # Ensure some space above the highest bar
ax.grid(axis="y", linestyle="--", alpha=0.6)
ax.tick_params(axis='x', rotation=15)  # Rotate x-axis labels for better readability

# Save the figure
plt.tight_layout()
plt.savefig("clustering_accuracy_ucf101_barchart_vertical.png", dpi=300)
plt.show()
