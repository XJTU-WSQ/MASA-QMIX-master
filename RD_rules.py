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
    sd_episode_reward, sd_episode_time_on_road, sd_episode_time_wait, _ = sd_rules_agent_wrapper(env.tasks_array)
    env.set_sd_episode_time_wait(sd_episode_time_wait)

    while not terminated and env.step_count < episode_limit:
        # print('step_count: ', step)

        actions = []
        env.renew_wait_time()
        for agent_id in range(n_agents):

            # 如果机器人被占用，则保持上一个step的任务分配决策结果；如果机器人空闲，则从可执行动作中选择一个动作去执行。
            env.renew_task_window(agent_id)
            avail_action = env.get_avail_actions(agent_id)
            action = random_choose_action(avail_action)
            # print("robot: ", agent_id, "avail_action: ", avail_action, "choose_action: ", action)
            if action != 7:
                env.has_chosen_action(action, agent_id)
            actions.append(action)
        reward, terminated = env.step(actions)
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


# 100个随机场景上的测试
# if __name__ == "__main__":
#     test_tasks = load_tasks_from_file('test_tasks.pkl')
#     episode_time_wait_list = []
#     for i in range(len(test_tasks)-99):
#         reward, time_on_road, time_wait, done = random_agent_wrapper(test_tasks[i])
#         print(reward)
#         print('task_id:', i, 'episode_time_on_road: ', time_on_road, 'episode_time_wait:', time_wait, "done:", done)
#         episode_time_wait_list.append(time_wait)
#     average_wait = sum(episode_time_wait_list)/len(episode_time_wait_list)
#     print('sd_algorithm average time_wait:', average_wait)


if __name__ == "__main__":
    tasks = load_tasks_from_file('task'
                                 '/fixed_tasks.pkl')
    print(tasks[0])
    rewards = []
    time_waits = []
    time_on_roads = []
    total_waits = []
    for i in range(1):
        reward, time_on_road, time_wait, done = random_agent_wrapper(tasks[0])
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


