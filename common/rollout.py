import copy
import numpy as np
import task_generator


class RolloutWorker:
    def __init__(self, env, agents, args):
        self.env = env
        self.agents = agents
        self.args = args
        self.n_actions = args.n_actions
        self.n_agents = args.n_agents
        self.state_shape = args.state_shape
        self.obs_shape = args.obs_shape
        self.episode_limit = args.episode_limit
        self.epsilon = args.epsilon
        self.anneal_epsilon = args.anneal_epsilon
        self.min_epsilon = args.min_epsilon

    def generate_episode(self, evaluate=False):
        # 初始化
        step = 0
        episode_reward = 0
        terminated = False
        epsilon = 0 if evaluate else self.epsilon
        # 用于存储每一步数据
        episode_data = []
        o, u, r, s, avail_u, u_onehot, terminate, padded = [], [], [], [], [], [], [], []
        self.agents.policy.init_hidden(1)
        self.env.reset()
        self.env.tasks_array = task_generator.generate_tasks()

        completed_tasks = 0
        conflicted_tasks = 0
        resolved_conflicts = 0
        total_wait_time = 0

        while not terminated and step < self.episode_limit:

            actions = np.zeros(self.args.n_agents)
            avail_actions = [[] for _ in range(self.args.n_agents)]
            actions_onehot = np.zeros((self.args.n_agents, self.args.n_actions))

            self.env.update_task_window()
            self.env.renew_wait_time()
            self.env.state4marl = self.env.get_state()
            self.env.update_observations()
            observations = [list(map(float, obs)) for obs in self.env.get_all_agent_obs()]
            actions_log = []

            # 收集当前任务和机器人信息
            waiting_tasks = []
            for task_index, task in enumerate(self.env.tasks_array):
                if self.env.tasks_allocated[task_index] == 0 and task[1] <= self.env.time:  # 未分配任务
                    waiting_tasks.append({
                        "task_index": task_index,
                        "request_pos": int(task[2]),
                        "target_pos": int(task[4]),
                        "priority": int(task[3]),
                        "service_time": int(task[5]),
                        "wait_time": int(self.env.time_wait[task_index])
                    })

            for agent_id in range(self.n_agents):

                avail_action = self.env.get_avail_actions(agent_id)
                action = self.agents.choose_action(
                    self.env.obs4marl[agent_id], agent_id, avail_action, epsilon
                )
                action = int(action)
                actions_log.append(int(action))

                # 生成 one-hot 动作
                action_onehot = np.zeros(self.args.n_actions)
                action_onehot[action] = 1
                actions[agent_id] = action
                actions_onehot[agent_id] = action_onehot
                avail_actions[agent_id] = avail_action

            actions = [int(a) for a in actions]

            # 统计任务冲突
            task_chosen = {}  # 用于记录当前 step 中任务的选择情况

            for agent_id, action in enumerate(actions):
                if action < self.env.task_window_size:  # 动作是一个有效任务
                    task_index = self.env.task_window[action][0]  # 获取任务索引
                    if task_index not in task_chosen:
                        task_chosen[task_index] = [agent_id]
                    else:
                        task_chosen[task_index].append(agent_id)

            # 统计冲突任务数量
            for task, agents in task_chosen.items():
                if len(agents) > 1:  # 如果有多个机器人选择了同一个任务
                    conflicted_tasks += 1
                    # 选择其中一个机器人执行任务，其余机器人被重新引导（假设选择第一个机器人）
                    resolved_conflicts += len(agents) - 1
                    for i in agents[1:]:
                        actions[i] = self.env.task_window_size  # 修改动作为 "不执行任务"
            reward, terminated = self.env.step(actions)

            # 保存观测、状态和动作信息
            o.append(copy.deepcopy(self.env.obs4marl))
            s.append(copy.deepcopy(self.env.state4marl))
            u.append(np.reshape(actions, [self.n_agents, 1]))
            u_onehot.append(actions_onehot)
            avail_u.append(avail_actions)
            # 进入env，执行一个step
            # 保存奖励，终止信息
            r.append([reward])
            terminate.append([terminated])
            padded.append([0.])
            # 添加 step_data 的构造逻辑
            step_data = {
                "step": step,
                "state": list(map(float, self.env.state4marl)),  # 确保 state 是原生类型
                "actions": actions_log,
                "reward": float(reward),
                "observations": observations,  # 观测已确保是原生类型
                "time": float(self.env.time),
                "waiting_tasks": waiting_tasks,
                "num_waiting_tasks": int(len(waiting_tasks)),
                "robots_info": [],
            }

            # 遍历所有机器人，收集机器人状态和任务信息
            for robot_id in range(self.env.robots.num_robots):
                robot_info = {
                    "robot_id": int(robot_id),  # 转换为 Python 原生整数
                    "position": list(map(float, self.env.robots.robot_pos[robot_id])),  # 转换为 Python 原生列表
                }
                if self.env.robots_state[robot_id] == 0:  # 空闲机器人
                    robot_info["status"] = "idle"
                    robot_info["current_task"] = None
                else:  # 忙碌机器人
                    task_info = self.env.robots.robots_tasks_info[robot_id]
                    robot_info["status"] = "busy"
                    robot_info["current_task"] = {
                        "task_index": int(task_info[0]),
                        "request_time": float(task_info[1]),
                        "site_id": int(task_info[2]),
                        "task_id": int(task_info[3]),
                        "destination_id": int(task_info[4]),
                    }
                step_data["robots_info"].append(robot_info)
            # 序列化数据
            episode_data.append(serialize_step_data(step_data))
            episode_reward += reward
            step += 1
        if self.args.epsilon_anneal_scale == 'episode':
            epsilon = epsilon - self.anneal_epsilon if epsilon > self.min_epsilon else self.min_epsilon

        # last obs
        o.append(copy.deepcopy(self.env.obs4marl))
        s.append(copy.deepcopy(self.env.state4marl))
        o_next = o[1:]
        s_next = s[1:]
        o = o[:-1]
        s = s[:-1]
        # get avail_action for last obs，because target_q needs avail_action in training
        avail_actions = []

        for agent_id in range(self.n_agents):
            avail_action = self.env.get_avail_actions(agent_id)
            avail_actions.append(avail_action)
        avail_u.append(avail_actions)
        avail_u_next = avail_u[1:]
        avail_u = avail_u[:-1]
        # if step < self.episode_limit，padding
        for i in range(step, self.episode_limit):
            o.append(np.zeros((self.n_agents, self.obs_shape)))
            u.append(np.zeros([self.n_agents, 1]))
            s.append(np.zeros(self.state_shape))
            r.append([0.])
            o_next.append(np.zeros((self.n_agents, self.obs_shape)))
            s_next.append(np.zeros(self.state_shape))
            u_onehot.append(np.zeros((self.n_agents, self.n_actions)))
            avail_u.append(np.zeros((self.n_agents, self.n_actions)))
            avail_u_next.append(np.zeros((self.n_agents, self.n_actions)))
            padded.append([1.])
            terminate.append([1.])  # 如果没有padding的情况下是没有terminal的

        episode = dict(o=o.copy(),
                       s=s.copy(),
                       u=u.copy(),
                       r=r.copy(),
                       avail_u=avail_u.copy(),
                       o_next=o_next.copy(),
                       s_next=s_next.copy(),
                       avail_u_next=avail_u_next.copy(),
                       u_onehot=u_onehot.copy(),
                       padded=padded.copy(),
                       terminated=terminate.copy()
                       )
        for key in episode.keys():
            episode[key] = np.array([episode[key]], dtype=object)

        # 统计等待时间
        total_wait_time = sum(self.env.time_wait)
        completed_tasks = sum(self.env.tasks_completed)
        average_wait_time = total_wait_time / len(self.env.tasks_array)

        episode_metrics = {
            "reward": episode_reward,
            "completed_tasks": completed_tasks,
            "conflicted_tasks": conflicted_tasks,
            "resolved_conflicts": resolved_conflicts,
            "average_wait_time": average_wait_time,
        }
        return episode, episode_reward, terminated, episode_data, episode_metrics

    def evaluate_episode(self, tasks):
        self.env.reset()
        self.env.tasks_array = tasks
        terminated = False
        step = 0
        epsilon = 0
        episode_reward = 0  # 累积奖励
        self.agents.policy.init_hidden(1)

        # 初始化统计量
        conflicted_tasks = 0
        resolved_conflicts = 0
        total_wait_time = 0
        completed_tasks = 0

        while not terminated and step < self.episode_limit:
            actions = np.zeros(self.args.n_agents)
            avail_actions = [[] for _ in range(self.args.n_agents)]
            actions_onehot = np.zeros((self.args.n_agents, self.args.n_actions))
            self.env.update_task_window()
            self.env.renew_wait_time()
            self.env.state4marl = self.env.get_state()
            self.env.update_observations()
            observations = [list(map(float, obs)) for obs in self.env.get_all_agent_obs()]

            task_chosen = {}  # 记录任务选择情况
            for agent_id in range(self.n_agents):
                avail_action = self.env.get_avail_actions(agent_id)
                action = self.agents.choose_action(
                    observations[agent_id], agent_id, avail_action, epsilon
                )
                action = int(action)

                # 统计任务冲突
                if action < self.env.task_window_size:
                    task_index = self.env.task_window[action][0]
                    if task_index not in task_chosen:
                        task_chosen[task_index] = [agent_id]
                    else:
                        task_chosen[task_index].append(agent_id)

                # 生成关于动作的 one-hot 向量
                action_onehot = np.zeros(self.args.n_actions)
                action_onehot[action] = 1
                actions[agent_id] = action
                actions_onehot[agent_id] = action_onehot
                avail_actions[agent_id] = avail_action

            # 统计冲突任务数量
            for task, agents in task_chosen.items():
                if len(agents) > 1:  # 多个机器人选择同一任务
                    conflicted_tasks += 1
                    resolved_conflicts += len(agents) - 1
                    for i in agents[1:]:
                        actions[i] = self.env.task_window_size  # 修改动作为 "不执行任务"

            # 执行动作
            actions = [int(a) for a in actions]
            reward, terminated = self.env.step(actions)
            episode_reward += reward
            step += 1

            # 更新统计量
        total_wait_time = sum(self.env.time_wait)
        completed_tasks = sum(self.env.tasks_completed)

        # 计算平均等待时间
        average_wait_time = total_wait_time / len(self.env.tasks_array)
        episode_metrics = {
            "reward": episode_reward,
            "completed_tasks": completed_tasks,
            "conflicted_tasks": conflicted_tasks,
            "resolved_conflicts": resolved_conflicts,
            "average_wait_time": average_wait_time,
        }

        return episode_reward, terminated, episode_metrics


def serialize_step_data(step_data):
    """
    将包含 ndarray 的 step_data 转换为 JSON 可序列化的数据结构。
    :param step_data: 原始 step_data 字典。
    :return: 可序列化的 step_data。
    """
    def convert(obj):
        if isinstance(obj, np.ndarray):  # 将 ndarray 转为 list
            return obj.tolist()
        elif isinstance(obj, list):  # 递归处理列表
            return [convert(item) for item in obj]
        elif isinstance(obj, dict):  # 递归处理字典
            return {key: convert(value) for key, value in obj.items()}
        else:
            return obj

    return convert(step_data)
