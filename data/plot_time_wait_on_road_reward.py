import numpy as np
import matplotlib.pyplot as plt

# 读取保存的奖励数据文件
data_r = np.genfromtxt('rewards_data.csv', delimiter=',')

i_epoch_1 = data_r[:, 0]
rewards = data_r[:, 1]

# 第一个子图
plt.figure(figsize=(15, 5))
plt.subplot(1, 2, 1)
plt.plot(i_epoch_1, rewards, label='Episode-Rewards')
plt.xlabel('Epoch')
plt.ylabel('Rewards')
plt.title('Reward Curve')
plt.grid(True)
plt.legend(loc='upper right', bbox_to_anchor=(1, 0.9))  # 调整标签位置

# 读取 time_wait 数据
data_t2 = np.genfromtxt('time_wait.csv', delimiter=',')

i_epoch_3 = data_t2[:, 0]
time_wait = data_t2[:, 1]

# 第三个子图
plt.subplot(1, 2, 2)
plt.plot(i_epoch_3, time_wait, label='time_wait')
plt.xlabel('Epoch')
plt.ylabel('time_wait')
plt.title('Time_wait Curve')
plt.grid(True)
plt.legend(loc='upper right')

plt.tight_layout()  # 调整子图之间的间距
plt.show()
