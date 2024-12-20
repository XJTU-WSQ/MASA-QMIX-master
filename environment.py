from abc import ABC
import gym
from gym.utils import seeding
from utils.sites import Sites
from utils.robots import Robots
from utils.tasks import Tasks
from utils.util import *
import task_generator


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
        self.task_window_size = 12
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
        self.obs4marl = [[0 for _ in range(self.obs_shape)] for _ in range(self.robots.num_robots)]
        self.robot_task_assignments = [[] for _ in range(self.robots.num_robots)]

    def reset(self):
        # 任务信息初始化
        self.task_window_size = 12
        self.time_wait = [0 for _ in range(len(self.tasks_array))]
        self.total_time_wait = 0
        self.tasks_completed = [0 for _ in range(len(self.tasks_array))]
        self.tasks_allocated = [0 for _ in range(len(self.tasks_array))]
        self.unallocated_tasks = set(range(len(self.tasks_array)))  # 未分配任务集合
        self.task_window = [[0 for _ in range(6)] for _ in range(self.task_window_size)]
        # 环境参数初始化
        self.time = 0.
        self.done = False
        self.state4marl = [0 for _ in range(len(self.get_state()))]
        # 机器人信息初始化
        self.robots.robot_pos = self.robots.robot_sites_pos
        self.robots_state = [0 for _ in range(self.robots.num_robots)]
        # 重新初始化 obs4marl
        self.obs4marl = [[0 for _ in range(self.obs_shape)] for _ in range(self.robots.num_robots)]
        self.robot_task_assignments = [[] for _ in range(self.robots.num_robots)]

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

    def get_state(self):
        """
        获取全局状态，包含机器人和任务的全局信息。
        """
        state = []
        max_pos_value = 110.0  # 假设坐标范围为 [0, 10]
        total_wait_time = max(self.total_time_wait, 1e-5)  # 避免分母为零

        # 添加机器人状态信息（归一化机器人位置）
        for robot_id in range(self.robots.num_robots):
            state.append(self.robots_state[robot_id])
            state.extend([p / max_pos_value for p in self.robots.robot_pos[robot_id]])

        # 添加任务信息
        waiting_tasks = [
            task for task_index, task in enumerate(self.tasks_array)
            if self.tasks_allocated[task_index] == 0 and task[1] <= self.time
        ]
        state.append(len(waiting_tasks) / self.task_window_size)  # 归一化任务数量

        for i in range(min(self.task_window_size, len(waiting_tasks))):
            task = waiting_tasks[i]
            task_index = task[0]
            state.append(task[4] / max_pos_value)  # 目标位置归一化
            state.append(self.time_wait[task_index] / total_wait_time)  # 动态归一化等待时间

        # 填充空位
        for _ in range(self.task_window_size - len(waiting_tasks)):
            state.extend([0, 0])

        return np.array(state, dtype=np.float32)

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

    def get_avail_actions(self, agent_id):
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

    def update_observations(self):
        """
        统一生成所有智能体的观测信息，缓存到 self.cached_obs4marl。
        """
        for agent_id in range(self.robots.num_robots):
            avail_action = self.get_avail_actions(agent_id)
            observation = self.get_agent_obs(agent_id, avail_action)
            self.obs4marl[agent_id] = observation

    def get_agent_obs(self, robot_id, avail_action):
        """
        获取当前机器人局部观测信息，基于当前任务窗特征和机器人与任务的关系。
        """
        observation = []
        robot_type = self.robots.robots_type_id[robot_id]
        speed = self.robots.speed[robot_type]
        robot_pos = self.robots.robot_pos[robot_id]

        # 1. 添加机器人自身信息（归一化机器人位置）
        max_pos_value = 110.0  # 假设坐标范围在 [0, 10]
        observation.extend([p / max_pos_value for p in robot_pos])  # 机器人位置归一化
        observation.append(float(self.robots_state[robot_id]))  # 是否空闲
        total_wait_time = max(self.total_time_wait, 1e-5)

        # 2. 添加任务信息
        for i, task in enumerate(self.task_window):
            if all(x == 0 for x in task) or avail_action[i] == 0:  # 无效任务
                observation.extend([0.0, 0.0])  # 无效任务占位
            else:
                task_index = task[0]
                target_pos = self.sites.sites_pos[task[4]]
                dis = distance(robot_pos, target_pos, speed)  # 机器人到任务的执行成本

                # 归一化特征
                normalized_dis = dis / max_pos_value  # 最大距离为 110
                normalized_wait_time = self.time_wait[task_index] / total_wait_time

                observation.extend([
                    normalized_dis,  # 当前机器人到任务的执行成本
                    normalized_wait_time,  # 任务累计等待时间
                ])
        return observation

    def get_all_agent_obs(self):
        """
        返回当前 step 内所有智能体的观测信息。
        """
        return self.obs4marl

    def step(self, actions):
        """
        执行所有智能体的动作，更新环境状态，并计算奖励。
        """
        time_span = 30  # 每个 step 的时间间隔
        max_wait_time = max(self.time_wait + [1e-5])  # 避免分母为零
        conflict_tasks = {}  # 冲突任务记录
        allocated_task_rewards = 0  # 当前 step 分配的任务奖励
        total_conflict_penalty = 0  # 冲突惩罚
        total_wait_penalty = 0  # 等待时间惩罚
        conflict_penalty = 0
        # 用于记录任务是否被重复选择
        task_chosen = {}

        # 更新忙碌机器人的状态
        for robot_id in range(self.robots.num_robots):
            if self.robots_state[robot_id] == 1:
                task_info = self.robots.robots_tasks_info[robot_id]
                finish = self.robots.renew_position(robot_id, task_info[3], task_info[2], task_info[4], time_span)
                if finish:
                    self.robots_state[robot_id] = 0
                    self.tasks_completed[task_info[0]] = 1

        # 遍历所有智能体，执行动作并更新任务分配状态
        for robot_id, action in enumerate(actions):
            if action < self.task_window_size:  # 有效动作
                task = self.task_window[action]
                task_index = task[0]

                if task_index in task_chosen:
                    task_chosen[task_index] += 1
                    total_conflict_penalty += 1
                else:
                    task_chosen[task_index] = 1

                if self.tasks_allocated[task_index] == 0:
                    time_on_road, total_time = self.robots.execute_task(robot_id, task)
                    self.robots_state[robot_id] = 1
                    self.tasks_allocated[task_index] = 1
                    self.total_time_wait += self.time_wait[task_index] + time_on_road
                    task_wait_time = self.time_wait[task_index]
                    normalized_allocated_reward = task_wait_time / max_wait_time
                    allocated_task_rewards += normalized_allocated_reward

        # 未分配任务的等待时间惩罚（归一化）
        unallocated_tasks = [
            task for task_index, task in enumerate(self.tasks_array)
            if self.tasks_allocated[task_index] == 0 and task[1] <= self.time
        ]
        for task in unallocated_tasks:
            task_wait_time = self.time_wait[task[0]]
            normalized_waiting_penalty = task_wait_time / max_wait_time
            total_wait_penalty += normalized_waiting_penalty

        # 冲突惩罚归一化
        for task_index, conflict_count in task_chosen.items():
            if conflict_count > 1:  # 存在冲突
                normalized_conflict_penalty = (conflict_count - 1) / self.robots.num_robots
                total_conflict_penalty += normalized_conflict_penalty

        # 计算总奖励
        reward = (
                allocated_task_rewards * 10  # 分配任务的奖励
                - total_wait_penalty * 0.1  # 等待时间惩罚
                - total_conflict_penalty * 5  # 冲突惩罚
        )

        self.time += time_span
        self.total_time_wait = sum(self.time_wait)
        if self.time > self.tasks_array[-1][1] and sum(self.tasks_completed) == len(self.tasks_completed):
            self.done = True
        return reward, self.done

    def get_env_info(self):
        """
        动态获取环境信息，包括 n_actions, n_agents, state_shape, obs_shape 和 episode_limit。
        """
        return {
            "n_actions": self.task_window_size + 1,  # 动作数量 = 任务窗大小 + 1
            "n_agents": self.robots.num_robots,  # 机器人数量
            "state_shape": len(self.get_state()),  # 全局状态向量的长度
            "obs_shape": self.obs_shape,  # 动态观测维度
            "episode_limit": 180
        }

