import re
import matplotlib.pyplot as plt

# Read loss values from the file
with open("loss.txt", "r") as f:
    loss_values = []

    for line in f:
        # Use regular expression to extract numeric values
        match = re.search(r'tensor\(([\d.]+)', line)
        if match:
            loss_values.append(float(match.group(1)))

# Plot the loss curve
plt.plot(loss_values, label='Loss')
plt.title('Loss Curve')
plt.xlabel('Iteration')
plt.ylabel('Loss')
plt.legend()
plt.show()

plt.show()
