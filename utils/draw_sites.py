from sites import Sites
import numpy as np
import matplotlib.pyplot as plt
from robots import Robots

plt.xlim((0, 60))
plt.ylim((0, 45))

s1 = Sites()
a = s1.rooms_pos

x_a, y_a = np.split(a, 2, 1)
plt.scatter(x_a, y_a, s=50, color='hotpink', alpha=0.5)

s2 = Robots()

b = s2.robot_sites_pos
colors = np.array(["red", "green", "black", "orange", "purple"])

print(len(b))
for i in range(len(b)):
    temp = np.array(b[i])
    x, y = np.split(temp, 2, 1)
    plt.scatter(x, y, s=50, c=colors[i], alpha=0.5)

c = s1.public_sites_pos
x_c, y_c = np.split(c, 2, 1)
plt.scatter(x_c, y_c, s=50, color='blue', alpha=0.5)

plt.show()
