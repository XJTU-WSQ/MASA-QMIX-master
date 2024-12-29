from abc import ABC
import gym
from gym.utils import seeding
from utils.sites import Sites
from utils.robots import Robots
from utils.tasks import Tasks
from utils.util import *
from task import task_generator


class ScheduleEnv(gym.Env, ABC):

    def __init__(self):
        # 实例化
        self.sites = Sites()
        self.robots = Robots()
        self.tasks = Tasks()
        # 机器人状态信息 : 1占用，0空闲
        self.robots_state = [0 for _ in range(self.robots.num_robots)]
        # 任务相关参数
        self.tasks_array = task_generator.generate_tasks()
        self.task_window_size = 10
        self.locked_tasks = [0] * self.task_window_size  # 记录任务锁定状态
        self.time_wait = [0 for _ in range(len(self.tasks_array))]
        self.total_time_wait = 0
        self.tasks_completed = [0 for _ in range(len(self.tasks_array))]
        self.tasks_allocated = [0 for _ in range(len(self.tasks_array))]
        self.unallocated_tasks = set(range(len(self.tasks_array)))  # 未分配任务集合
        self.task_window = np.array([[0 for _ in range(6)] for _ in range(self.task_window_size)])
        # 环境相关参数
        self.cached_obs4marl = None
        self.time = 0.
        self.done = False
        self.state4marl = [0 for _ in range(len(self.get_state()))]
        # 动态获取 obs_shape
        temp_avail_action = [0] * (self.task_window_size + 1)
        temp_obs = self.get_agent_obs(0, temp_avail_action)
        self.obs_shape = len(temp_obs)  # 动态观测维度
        # 初始化 obs4marl
        self.obs4marl = np.zeros((self.robots.num_robots, self.obs_shape), dtype=np.float32)
        self.robot_task_assignments = [[] for _ in range(self.robots.num_robots)]

    def reset(self):
        """
        环境重置，初始化所有参数。
        """
        # 任务信息初始化
        self.task_window_size = 10
        self.time_wait = [0 for _ in range(len(self.tasks_array))]
        self.total_time_wait = 0
        self.tasks_completed = [0 for _ in range(len(self.tasks_array))]
        self.tasks_allocated = [0 for _ in range(len(self.tasks_array))]
        self.unallocated_tasks = set(range(len(self.tasks_array)))
        self.task_window = [[0 for _ in range(6)] for _ in range(self.task_window_size)]

        # 环境参数初始化
        self.time = 0.0
        self.done = False
        self.state4marl = [0 for _ in range(len(self.get_state()))]

        # 机器人信息初始化
        self.robots.robot_pos = self.robots.robot_sites_pos
        self.robots_state = [0 for _ in range(self.robots.num_robots)]

        # 初始化 obs4marl
        self.obs4marl = np.zeros((self.robots.num_robots, self.obs_shape), dtype=np.float32)
        self.robot_task_assignments = [[] for _ in range(self.robots.num_robots)]
        self.update_task_window()

        return self.get_obs(), self.get_state()

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def renew_wait_time(self):
        """
        优化后的 renew_wait_time：只遍历未分配任务集合。
        """
        tasks_to_remove = []
        for task_index in self.unallocated_tasks:
            if self.tasks_array[task_index][1] <= self.time:
                self.time_wait[task_index] = self.time - self.tasks_array[task_index][1]
            else:
                break  # 任务数组有序，可以提前退出
            if self.tasks_allocated[task_index] == 1:
                tasks_to_remove.append(task_index)

        # 从未分配任务集合中移除已分配任务
        for task_index in tasks_to_remove:
            self.unallocated_tasks.remove(task_index)

    def update_task_window(self):
        """
        更新任务窗：只在每个 step 开始时生成一次任务窗。
        """
        m = 0
        self.task_window = [[0 for _ in range(6)] for _ in range(self.task_window_size)]
        for task_index in range(len(self.tasks_array)):
            if self.tasks_array[task_index][1] <= self.time and self.tasks_allocated[task_index] == 0:
                self.task_window[m] = self.tasks_array[task_index].tolist()
                m += 1
                if m == self.task_window_size:
                    break
            if self.tasks_array[task_index][1] > self.time:
                break

    def get_state(self):
        """
        获取全局状态，包含机器人和任务的全局信息。
        """
        state = []
        max_pos_value = 110.0  # 坐标范围为 [0, 110]
        max_wait_time = max(self.time_wait) if max(self.time_wait) > 0 else 1e-5

        # 添加机器人状态信息（机器人归一化位置信息，以及机器人状态信息）
        for robot_id in range(self.robots.num_robots):
            state.append(self.robots_state[robot_id])
            state.append(self.robots.robot_pos[robot_id][0] / max_pos_value)
            state.append(self.robots.robot_pos[robot_id][1] / max_pos_value)

        # 添加任务信息（任务的归一化位置信息，任务的归一化等待时长）
        waiting_tasks = [
            task for task_index, task in enumerate(self.tasks_array)
            if self.tasks_allocated[task_index] == 0 and task[1] <= self.time
        ]

        for i in range(min(self.task_window_size, len(waiting_tasks))):
            task = waiting_tasks[i]
            [task_index, _, site_id, task_type, destination_id, _] = task
            if task_type == 2 or task_type == 3:
                destination_pos = self.sites.sites_pos[destination_id]
            else:
                destination_pos = self.sites.sites_pos[site_id]
            state.append(destination_pos[0] / max_pos_value)  # x 坐标归一化
            state.append(destination_pos[1] / max_pos_value)  # y 坐标归一化
            state.append(self.time_wait[task_index] / max_wait_time)  # 动态归一化等待时间

        # 填充空位
        for _ in range(self.task_window_size - len(waiting_tasks)):
            state.extend([0., 0., 0.])

        return np.array(state, dtype=np.float32)

    def get_obs(self):
        """
        获取所有智能体的观测信息。
        """
        for agent_id in range(self.robots.num_robots):
            avail_action = self.get_avail_agent_actions(agent_id)
            observation = self.get_agent_obs(agent_id, avail_action)
            self.obs4marl[agent_id] = observation
        return self.obs4marl

    def get_agent_obs(self, robot_id, avail_action):
        """
        获取单个机器人的局部观测，包含机器人自身信息、任务窗口信息。
        """
        observation = []
        robot_pos = self.robots.robot_pos[robot_id]
        max_pos_value = 110.0  # 假设坐标最大值
        max_distance = 160.0  # 假设环境中最大曼哈顿距离
        max_wait_time = max(self.time_wait) if max(self.time_wait) > 0 else 1e-5  # 避免除以零
        max_task_window_size = self.task_window_size  # 假设任务窗口大小固定

        # 1. 添加机器人自身信息(状态信息和归一化位置信息)
        normalized_robot_pos = [robot_pos[0] / max_pos_value, robot_pos[1] / max_pos_value]
        observation.append(self.robots_state[robot_id] )  # 状态信息
        observation.extend(normalized_robot_pos)  # 归一化位置信息

        # 2. 添加任务窗口信息（仅包含可执行任务，归一化等待时间和归一化距离）
        task_features = []
        for i, task in enumerate(self.task_window):
            if all(x == 0 for x in task) or avail_action[i] == 0:  # 占位符任务或不可执行
                task_features.append([0.0, 0.0])  # 占位符
            else:
                [task_index, _, site_id, task_type, destination_id, _] = task
                if task_type == 2 or task_type == 3:
                    destination_pos = self.sites.sites_pos[destination_id]
                else:
                    destination_pos = self.sites.sites_pos[site_id]
                dis = (abs(robot_pos[0] - destination_pos[0]) + abs(robot_pos[1] - destination_pos[1])) / max_distance
                wait_time = self.time_wait[task_index] / max_wait_time  # 等待时间归一化
                task_features.append([dis, wait_time])

        # 如果任务数量少于任务窗口大小，填充占位符
        while len(task_features) < max_task_window_size:
            task_features.append([0.0, 0.0])

        # 截取到固定任务窗口大小
        task_features = task_features[:max_task_window_size]

        # 将任务特征展平为单一列表
        for feature in task_features:
            observation.extend(feature)

        return np.array(observation, dtype=np.float32)

    def get_avail_agent_actions(self, agent_id):
        """
        获取当前机器人可执行的任务，并对无效任务动态掩码，返回动作掩码和惩罚权重。
        """
        avail_actions = [0] * (self.task_window_size + 1)  # 初始化动作掩码，+1 表示 "不执行任务"
        avail_actions[-1] = 1  # 默认 "不执行任务" 动作可选

        # 如果机器人忙碌，直接返回
        if self.robots_state[agent_id] == 1:
            return avail_actions

        # 机器人技能
        robot_skill = self.robots.get_skills(agent_id)

        # 遍历任务窗，检查任务是否有效
        for j, task in enumerate(self.task_window):
            if task == [0, 0, 0, 0, 0, 0]:  # 判断任务是否为占位符
                continue

            task_skill_list = self.tasks.required_skills[task[3]]  # 获取任务需求技能

            # 判断技能是否匹配
            if all(rs >= ts for rs, ts in zip(robot_skill, task_skill_list)):
                avail_actions[j] = 1  # 设置任务为可选
            else:
                avail_actions[j] = 0  # 机器人技能不匹配，任务不可选
        return avail_actions

    def step(self, actions):
        """
        执行智能体的动作，更新环境状态，并计算综合奖励。
        """
        time_step = 30  # 每个 step 的时间间隔
        task_rewards = 0  # 任务完成的正奖励
        total_service_cost_penalty = 0  # 服务成本的总惩罚
        task_allocated_reward = 1   # 每完成一个任务分配的固定奖励
        service_cost_penalty_weight = -0.02  # 服务成本惩罚的权重
        wait_penalty_weight = -0.05  # 等待一个任务就进行一个惩罚
        conflict_count = 0  # 记录冲突数量
        # 用于记录任务分配情况，检测冲突
        task_allocation = {}
        # 更新所有忙碌机器人状态
        for robot_id in range(self.robots.num_robots):
            if self.robots_state[robot_id] == 1:  # 忙碌机器人
                task_info = self.robots.robots_tasks_info[robot_id]
                finished = self.robots.renew_position(robot_id, task_info[3], task_info[2], task_info[4], time_step)
                if finished:
                    self.robots_state[robot_id] = 0
                    self.tasks_completed[task_info[0]] = 1  # 标记任务完成

        # 遍历所有机器人动作，分配任务并记录任务分配
        for robot_id, action in enumerate(actions):
            if action < self.task_window_size:  # 如果动作合法
                task_index = action
                # 检测是否发生冲突
                if task_index in task_allocation:
                    task_allocation[task_index].append(robot_id)
                else:
                    task_allocation[task_index] = [robot_id]
        # 处理冲突任务
        for task_index, agents in task_allocation.items():
            if len(agents) == 1:  # 如果没有冲突
                agent_id = agents[0]
                task = self.task_window[task_index]
                time_on_road, total_time = self.robots.execute_task(agent_id, task)
                self.robots_state[agent_id] = 1
                self.tasks_allocated[task[0]] = 1
                task_rewards += task_allocated_reward
                total_service_cost_penalty += time_on_road * service_cost_penalty_weight
            if len(agents) > 1:  # 如果任务发生冲突
                conflict_count += 1  # 冲突数量 +1
                chosen_agent = np.random.choice(agents)  # 随机选择一个机器人执行任务
                for agent_id in agents:
                    if agent_id == chosen_agent:
                        # 确保随机选择的机器人执行任务
                        task = self.task_window[task_index]
                        time_on_road, total_time = self.robots.execute_task(agent_id, task)
                        self.robots_state[agent_id] = 1
                        self.tasks_allocated[task[0]] = 1
                        task_rewards += task_allocated_reward  # 成功分配任务奖励
                        total_service_cost_penalty += time_on_road * service_cost_penalty_weight

        step_wait_num = sum(1 for task in self.tasks_array if self.tasks_allocated[task[0]] == 0 and task[1] <= self.time)
        total_wait_penalty = wait_penalty_weight * step_wait_num

        # 更新时间步
        self.time += time_step
        self.total_time_wait = sum(self.time_wait)  # 累计等待时间
        done = self.time > self.tasks_array[-1][1] and sum(self.tasks_completed) == len(self.tasks_array)

        # 综合奖励
        total_reward = task_rewards + total_wait_penalty + total_service_cost_penalty

        info = {
            "robots_state": self.robots_state,
            "task_window": self.task_window,
            "conflict_count": conflict_count,
            "task_rewards": task_rewards,
            "total_wait_penalty": total_wait_penalty,
            "total_service_cost_penalty": total_service_cost_penalty,
            "done": done
        }
        return total_reward, done, info

    def get_env_info(self):
        """
        动态获取环境信息，包括 n_actions, n_agents, state_shape, obs_shape 和 episode_limit。
        """
        return {
            "n_actions": self.task_window_size + 1,  # 动作数量 = 任务窗大小 + 1
            "n_agents": self.robots.num_robots,  # 机器人数量
            "state_shape": len(self.get_state()),  # 全局状态向量的长度
            "obs_shape": self.obs_shape,  # 动态观测维度
            "episode_limit": 170
        }

