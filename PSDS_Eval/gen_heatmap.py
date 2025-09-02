import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# Define the confusion matrix
# cm = np.array([
#     [0.4256, 0.0869, 0.0606, 0.4269],
#     [0.0543, 0.3804, 0.0414, 0.5239],
#     [0.1739, 0.1840, 0.2879, 0.3542],
#     [0.0370, 0.0879, 0.0162, 0.8590]
# ])



cm = np.array([[0.4119, 0.0915, 0.0652, 0.4315],
             [0.0608, 0.3609, 0.0479, 0.5304],
             [0.1797, 0.1898, 0.2706, 0.3600],
             [0.0423, 0.0932, 0.0215, 0.8429]]
              )


# Map the numeric labels to category names in the order used
labels = ['rhonchi', 'wheeze', 'stridor', 'crackle']

# Plot the heatmap with category names on the axes
plt.figure(figsize=(8, 6))
sns.heatmap(
    cm,
    annot=True,
    fmt=".4f",
    xticklabels=labels,
    yticklabels=labels,
    cmap="Blues"
)
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.title("Confusion Matrix Heatmap")
plt.tight_layout()
plt.show()
