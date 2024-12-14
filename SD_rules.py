# 优先选择离自己最近的可执行动作
from environment import ScheduleEnv
from test_task import load_tasks_from_file

TASK_INFO = ["紧急事件", "去卫生间（快速运输任务）", "送餐", "情感陪护", "移动辅助任务", "私人物品配送", "康复训练"]

LOCATIONS = ["1号房间", "2号房间", "3号房间", "4号房间", "5号房间", "6号房间",
             "7号房间", "8号房间", "9号房间", "10号房间", "11号房间", "12号房间",
             "13号房间", "14号房间", "15号房间", "16号房间", "17号房间", "18号房间",
             "餐厅", "室外活动区南", "室外活动区北", "活动室", "浴室", "值班室", "西北卫生间", "东南卫生间"]

TASK_TYPES = {
    "general": [1, 2, 3, 4, 5, 6],
    "canteen": [1, 3, 4, 5, 6],
    "toilet": [4, 5, 6]
}

TASK_PROBABILITIES = {
    "general": [0.3, 0.1, 0.1, 0.3, 0.1, 0.1],
    "canteen": [0.4, 0.2, 0.1, 0.2, 0.1],
    "toilet": [0.6, 0.3, 0.1]
}


def sd_rules_agent_wrapper(tasks_array):

    env = ScheduleEnv()
    env_info = env.get_env_info()
    n_actions = env_info["n_actions"]
    n_agents = env_info["n_agents"]
    episode_limit = env_info["episode_limit"]
    env.reset()
    env.tasks_array = tasks_array
    # print(tasks)
    terminated = False
    step = 0  # 累积奖励
    episode_reward = 0

    while not terminated and env.step_count < episode_limit:
        # print('step_count: ', step)
        step += 1
        actions = []
        env.renew_wait_time()
        # for agent_id in range(n_agents):
        for agent_id in range(n_agents):
            env.renew_task_window(agent_id)
            avail_action = env.get_avail_actions(agent_id)
            action = env.choose_sd_action(avail_action, agent_id)
            # print("robot: ", agent_id, "avail_action: ", avail_action, "choose_action: ", action)
            if action != 7:
                env.has_chosen_action(action, agent_id)
            actions.append(action)
        reward, terminated = env.step(actions)
        episode_reward += reward
    episode_time_on_road, episode_time_wait = env.get_time()

    return episode_reward, episode_time_on_road, episode_time_wait, terminated


# # 100个随机场景上的测试
if __name__ == "__main__":
    test_tasks = load_tasks_from_file('task/test_tasks.pkl')
    episode_time_wait_list = []
    # for i in range(len(test_tasks)-99):
    #     reward, time_on_road, time_wait, done = sd_rules_agent_wrapper(test_tasks[i])
    #     print('task_id:', i, 'episode_time_on_road: ', time_on_road, 'episode_time_wait:', time_wait, "done:", done)
    #     episode_time_wait_list.append(time_wait)
    reward, time_on_road, time_wait, done = sd_rules_agent_wrapper(test_tasks[0])
    print('episode_time_on_road: ', time_on_road, 'episode_time_wait:', time_wait, "done:", done)
    episode_time_wait_list.append(time_wait)
    average_wait = sum(episode_time_wait_list)/len(episode_time_wait_list)
    print('sd_algorithm average time_wait:', average_wait)

    all_task = test_tasks[0]
    # 假设 all_task 是你的任务数据，LOCATIONS 和 TASK_INFO 是你已经定义好的字典。

    with open('tasks', 'w') as f:
        # 在开始时写入表头（如果需要）
        f.write(
            "Task index  | Time  | Location          | Task Type                  | Target            | Duration\n"
            "------------------------------------------------------------\n"
        )

        # 然后逐行写入数据
        for i in range(len(all_task)):
            task = all_task[i]

            task_info = {
                'Task index': task[0],
                'Time': task[1],
                'Location': LOCATIONS[task[2]],
                'Task Type': TASK_INFO[task[3]],
                'Target': LOCATIONS[task[4]],
                'Duration': task[5],
            }

            # 定义格式化字符串，确保每列的宽度和对齐方式一致
            msg = (
                "{:<12} | {:<5} | {:<18} | {:<30} | {:<18} | {:>8}"  # 设置每列的宽度与对齐
            ).format(
                task_info['Task index'],  # Task index
                task_info['Time'],  # Time
                task_info['Location'],  # Location
                task_info['Task Type'],  # Task Type
                task_info['Target'],  # Target
                task_info['Duration']  # Duration (right-aligned)
            )

            # 写入每一行数据
            f.write(msg + "\n")

# if __name__ == "__main__":
#     tasks = load_tasks_from_file('fixed_tasks.pkl')
#     reward, time_on_road, time_wait, done = sd_rules_agent_wrapper((tasks[0]))
#     print('episode_time_on_road: ', time_on_road, 'episode_time_wait:', time_wait, "done:", done)


