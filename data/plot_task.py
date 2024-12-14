import numpy as np
import matplotlib.pyplot as plt


def load_data(prefix):
    episode_rewards = np.genfromtxt(f'{prefix}_episode_rewards.csv', delimiter=',')
    episode_time_wait = np.genfromtxt(f'{prefix}_episode_time_wait.csv', delimiter=',')

    return episode_rewards, episode_time_wait


# 读取数据
algorithm1_episode_rewards, algorithm1_episode_time_wait = load_data('qmix')
algorithm2_episode_rewards, algorithm2_episode_time_wait = load_data('sd')
algorithm3_episode_rewards, algorithm3_episode_time_wait = load_data('rd')

# 绘制图表
fig, axes = plt.subplots(2, 4, figsize=(24, 12))
plt.subplots_adjust(left=0.1, right=0.9, top=0.9, bottom=0.1, hspace=0.5)

fig.suptitle('four Task Evaluate Over Epochs')

# 绘制 episode_rewards
for i in range(4):
    axes[0, i].plot(algorithm1_episode_rewards[i, :], label=f'Qmix Reward')
    axes[0, i].plot(algorithm2_episode_rewards[i, :], label=f'SD-FIFO Reward')
    axes[0, i].plot(algorithm3_episode_rewards[i, :], label=f'Random Reward')
    axes[0, i].set_title(f'Reward {i+1}')
    axes[0, i].set_xlabel('Epoch*20')
    axes[0, i].set_ylabel('Reward')
    axes[0, i].legend()

# 绘制 episode_time_wait
for i in range(4):
    axes[1, i].plot(algorithm1_episode_time_wait[i, :], label=f'Qmix Time Wait')
    axes[1, i].plot(algorithm2_episode_time_wait[i, :], label=f'SD-FIFO Time Wait')
    axes[1, i].plot(algorithm3_episode_time_wait[i, :], label=f'Random Time Wait')
    axes[1, i].set_title(f'Time Wait {i+1}')
    axes[1, i].set_xlabel('Epoch*20')
    axes[1, i].set_ylabel('Time Wait')
    axes[1, i].legend()

# 在每一列上加上任务编号
for i in range(4):
    axes[1, i].annotate(f'Task {i+1}', (0.5, -0.2), xycoords='axes fraction', ha='center', va='center', fontsize=8)

plt.tight_layout(rect=[0, 0, 1, 0.96], h_pad=1)
plt.show()



