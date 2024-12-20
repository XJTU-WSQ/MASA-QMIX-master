# 随机决策函数，用于测试环境
from environment import ScheduleEnv
import numpy as np
from test_task import load_tasks_from_file
from SD_rules import sd_rules_agent_wrapper


def random_agent_wrapper(tasks_array):

    env = ScheduleEnv()
    env_info = env.get_env_info()
    n_actions = env_info["n_actions"]
    n_agents = env_info["n_agents"]
    episode_limit = env_info["episode_limit"]
    env.reset()
    env.tasks_array = tasks_array
    # print(env.tasks_array)
    terminated = False
    step = 0  # 累积奖励
    episode_reward = 0

    while not terminated and step < episode_limit:
        # print('step_count: ', step)

        actions = []
        env.renew_wait_time()
        for agent_id in range(n_agents):

            # 如果机器人被占用，则保持上一个step的任务分配决策结果；如果机器人空闲，则从可执行动作中选择一个动作去执行。
            avail_action = env.get_avail_actions(agent_id)
            action = random_choose_action(avail_action)
            # print("robot: ", agent_id, "avail_action: ", avail_action, "choose_action: ", action)
            actions.append(action)
        terminated = env.step(actions)
        step += 1
        episode_reward += reward
    episode_time_on_road, episode_time_wait = env.get_time()
    # print('episode_time_on_road: ', episode_time_on_road, 'episode_time_wait:', episode_time_wait, "done:",
    #       terminated)
    return episode_reward, episode_time_on_road, episode_time_wait, terminated


def random_choose_action(avail_action):
    # 找到值为 1 的索引
    indices = np.where(np.array(avail_action) == 1)[0]
    action = np.random.choice(indices)
    return action


if __name__ == "__main__":
    test_tasks = load_tasks_from_file('task/test_tasks.pkl')
    print(test_tasks[0])
    rewards = []
    time_waits = []
    time_on_roads = []
    total_waits = []
    for i in range(1):
        reward, time_on_road, time_wait, done = random_agent_wrapper(test_tasks[0])
        rewards.append(reward)
        time_on_roads.append(time_on_road)
        time_waits.append(time_wait)
        print("num:", i, 'episode_reward: ', reward, 'episode_time_on_road: ', time_on_road, 'episode_time_wait:', time_wait, "done:", done)

    # 使用argsort获取排序后的索引数组
    sorted_indices = np.argsort(rewards)[::-1]  # [::-1]表示降序排列

    # 根据排序后的索引对两个列表进行重新排列
    sorted_rewards = [rewards[i] for i in sorted_indices]
    sorted_time_waits = [time_waits[i] for i in sorted_indices]
    combined_list = list(zip(sorted_rewards, sorted_time_waits))
    print("Combined List:")
    for row in combined_list:
        print(row)
    print("min_reward:", max(rewards), "min_time_on_road:", min(time_on_roads), "min_time_wait:", min(time_waits))


