from abc import ABC
from utils.sites import Sites
from utils.robots import Robots, get_enable_flag
from utils.tasks import Tasks
import gym
from gym.utils import seeding
from utils.util import *
import task_generator_fixed


def is_all_zero(matrix):
    for row in matrix:
        for element in row:
            if element != 0:
                return False
    return True


def reward_function(wait_time, time_on_road):
    reward = -(wait_time + time_on_road)*0.01
    return reward


class ScheduleEnv(gym.Env, ABC):

    def __init__(self):
        # 实例化
        self.sites = Sites()
        self.robots = Robots()
        self.tasks = Tasks()
        # 任务相关参数
        # self.time_total_wait = 0
        self.tasks_array = task_generator_fixed.generate_tasks()
        self.time_wait = [0 for _ in range(len(self.tasks_array))]
        self.total_time_wait = 0
        self.time_on_road = [0 for _ in range(self.robots.num_robots)]
        self.total_time_on_road = 0
        self.tasks_completed = [0 for _ in range(len(self.tasks_array))]
        self.tasks_allocated = [0 for _ in range(len(self.tasks_array))]
        self.task_window = np.array([[0 for _ in range(6)] for _ in range(7)])
        # 环境相关参数
        self.time = 0.
        self.done = False
        self.step_count = 0
        self.state4marl = [0 for _ in range(17)]
        self.step_reward = []
        self.baseline = 0
        # 机器人状态信息 : 1占用，0空闲
        self.robots_state = [0 for _ in range(self.robots.num_robots)]
        self.robots.robots_type_available_num = [self.robots.n_walking, self.robots.n_wheelchair,
                                                 self.robots.n_delivery, self.robots.n_company,
                                                 self.robots.n_private_delivery]
        self.obs4marl = [[0 for _ in range(14)] for _ in range(self.robots.num_robots)]
        self.robot_task_assignments = [[] for _ in range(self.robots.num_robots)]

    def reset(self):
        # 任务信息初始化
        self.time_wait = [0 for _ in range(len(self.tasks_array))]
        self.total_time_wait = 0
        self.time_on_road = [0 for _ in range(self.robots.num_robots)]
        self.total_time_on_road = 0
        self.tasks_completed = [0 for _ in range(len(self.tasks_array))]
        self.tasks_allocated = [0 for _ in range(len(self.tasks_array))]
        self.task_window = [[0 for _ in range(6)] for _ in range(7)]
        # 环境参数初始化
        self.time = 0.
        self.done = False
        self.step_count = 0
        self.state4marl = [0 for _ in range(17)]
        self.step_reward = []
        self.baseline = 0
        # 机器人信息初始化
        self.robots.robot_pos = self.robots.robot_sites_pos
        self.robots_state = [0 for _ in range(self.robots.num_robots)]
        self.robots.robots_type_available_num = [self.robots.n_walking, self.robots.n_wheelchair,
                                                 self.robots.n_delivery, self.robots.n_company,
                                                 self.robots.n_private_delivery]
        self.obs4marl = [[0 for _ in range(14)] for _ in range(self.robots.num_robots)]
        self.robot_task_assignments = [[] for _ in range(self.robots.num_robots)]

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def renew_wait_time(self):
        # index = 0
        for task_index in range(len(self.tasks_array)):
            if self.tasks_array[task_index][1] <= self.time and self.tasks_allocated[task_index] == 0:
                self.time_wait[task_index] = self.time - self.tasks_array[task_index][1]
            if self.tasks_array[task_index][1] > self.time:
                break
        # print('time_wait: ', self.time_wait)

    def get_state(self):
        self.state4marl[0:17] = self.robots_state
        # self.state4marl[17] = self.waiting_task_num
        return self.state4marl

    def renew_task_window(self, agent_id):
        m = 0
        self.task_window = [[0 for _ in range(6)] for _ in range(7)]
        robot_type = self.robots.robots_type_id[agent_id]
        for task_index in range(len(self.tasks_array)):
            if self.tasks_array[task_index][1] <= self.time and self.tasks_allocated[task_index] == 0 and \
                    get_enable_flag(self.tasks_array[task_index][3], robot_type):
                self.task_window[m] = self.tasks_array[task_index].tolist()
                m += 1
                if m == 7:
                    break
            if self.tasks_array[task_index][1] > self.time:
                break
        # print('task_window: ', self.task_window)

    # 函数功能：机器人的技能集，与任务窗任务的技能集比较，得到可以选择的动作。
    def get_avail_actions(self, agent_id):
        avail_actions = [0, 0, 0, 0, 0, 0, 0, 1]
        if self.robots_state[agent_id] == 1 or is_all_zero(self.task_window):
            return [0, 0, 0, 0, 0, 0, 0, 1]
        # 任务集和机器人技能集做匹配
        tasks_skill_list = self.tasks.get_tasks_required_skill_list(self.task_window)
        robot_skill = self.robots.get_skills(agent_id)
        for j, task_skill_list in enumerate(tasks_skill_list):
            flag = 0
            for i in range(len(robot_skill)):
                if task_skill_list[i] == 1 and robot_skill[i] == 1:
                    flag = 1
                elif task_skill_list[i] == 1 and robot_skill[i] == 0:
                    flag = 0
                    break
            if flag == 1:
                avail_actions[j] = 1
                avail_actions[7] = 0
        return avail_actions

    def get_agent_obs(self, robot_id, avail_action):
        # 获取机器人的任务类型，以及对应类型机器人的移动速度
        if self.robots_state[robot_id] == 1:
            self.obs4marl[robot_id] = [0 for _ in range(14)]
            # print('robot_id: ', robot_id, 'obs: ', self.obs4marl[robot_id])
            return self.obs4marl[robot_id]
        robot_type = self.robots.robots_type_id[robot_id]
        speed = self.robots.speed[robot_type]
        for i, task in enumerate(self.task_window):
            # all(element == 0 for element in task)
            if avail_action[i] == 0:
                self.obs4marl[robot_id][2*i: 2*i + 2] = [0, 0]
            else:
                [task_index, _, request_id, task_type, destination_id, service_time] = task
                if task_type == 2 or task_type == 5:
                    first_place = destination_id
                else:
                    first_place = request_id
                dis = distance(self.robots.robot_pos[robot_id], self.sites.sites_pos[first_place], speed)
                time_wait = self.time_wait[task_index]
                self.obs4marl[robot_id][2*i: 2*i + 2] = [dis, time_wait]
        # print('robot_id: ', robot_id, 'obs: ', self.obs4marl[robot_id])
        return self.obs4marl[robot_id]

    # def get_agent_obs(self, robot_id, avail_action):
    #     # 获取机器人的任务类型，以及对应类型机器人的移动速度
    #     if self.robots_state[robot_id] == 1:
    #         self.obs4marl[robot_id] = [0 for _ in range(7)]
    #         # print('robot_id: ', robot_id, 'obs: ', self.obs4marl[robot_id])
    #         return self.obs4marl[robot_id]
    #     robot_type = self.robots.robots_type_id[robot_id]
    #     speed = self.robots.speed[robot_type]
    #     for i, task in enumerate(self.task_window):
    #         # all(element == 0 for element in task)
    #         if avail_action[i] == 0:
    #             self.obs4marl[robot_id][i] = 0
    #         else:
    #             [task_index, _, request_id, task_type, destination_id, service_time] = task
    #             if task_type == 2 or task_type == 5:
    #                 first_place = destination_id
    #             else:
    #                 first_place = request_id
    #             arrive_time = distance(self.robots.robot_pos[robot_id], self.sites.sites_pos[first_place], speed)
    #             time_wait = self.time_wait[task_index]
    #             self.obs4marl[robot_id][i] = arrive_time + time_wait
    #     # print('robot_id: ', robot_id, 'obs: ', self.obs4marl[robot_id])
    #     return self.obs4marl[robot_id]

    def has_chosen_action(self, action_id, robot_id):
        task = self.task_window[action_id]
        [task_index, requests_time, site_id, task_id, destination_id, service_time] = task
        time_on_road, total_time = self.robots.execute_task(robot_id, task)
        self.time_on_road[robot_id] = time_on_road
        # 更新机器人的占用状态，更新对应类型可使用机器人个数
        self.robots_state[robot_id] = 1
        robot_type = self.robots.robots_type_id[robot_id]
        self.robots.robots_type_available_num[robot_type] -= 1
        # 更新任务分配情况
        self.tasks_allocated[task_index] = 1
        wait_time = self.time_wait[task_index]
        reward = reward_function(wait_time, time_on_road)
        self.step_reward.append(reward)
        assignment_info = {
            'task_index': task_index,
            'requests_time': requests_time,
            'allocate_time': self.time,
            'wait_time': self.time_wait[task_index],
            'site_id': site_id,
            'task_id': task_id,
            'destination_id': destination_id,
            'service_time': service_time,
            'time_on_road': time_on_road,
            'total_time': total_time,
            'reward': reward
        }
        self.robot_task_assignments[robot_id].append(assignment_info)

    def step(self, actions):

        self.step_count += 1
        time_span = 60

        # print('step_count: ', self.step_count)
        # print("robot_state", self.robots_state)
        # print('time: ', self.time)
        # print('actions: ', actions)
        for robot_id in range(self.robots.num_robots):
            # 如果机器人正忙，更新机器人坐标
            if self.robots_state[robot_id] == 1:
                [task_index, requests_time, site_id, task_id, destination_id] = self.robots.robots_tasks_info[robot_id]
                finish = self.robots.renew_position(robot_id, task_id, site_id, destination_id, time_span)
                if finish:
                    # 更新机器人状态
                    self.robots_state[robot_id] = 0
                    robot_type = self.robots.robots_type_id[robot_id]
                    self.robots.robots_type_available_num[robot_type] += 1
                    self.tasks_completed[task_index] = 1

        # 更新时间
        self.time += time_span
        self.total_time_on_road += sum(self.time_on_road)
        self.total_time_wait = sum(self.time_wait) + self.total_time_on_road
        if self.time > self.tasks_array[-1][1] and sum(self.tasks_allocated) == len(self.tasks_allocated):
            self.done = True

        reward = 0
        if self.done:
            reward = (self.baseline - self.total_time_wait)*0.5
            # print(self.baseline)
        else:
            reward = sum(self.step_reward)
            # self.save_task_assignments_to_file()
        # print("reward:", reward)

        self.time_on_road = [0 for i in range(self.robots.num_robots)]
        self.step_reward = []
        return reward, self.done

    def get_time(self):
        return self.total_time_on_road, self.total_time_wait

    def get_env_info(self):
        return {
            "n_actions": 8,
            "n_agents": self.robots.num_robots,
            "state_shape": len(self.get_state()),
            "obs_shape": 14,
            "episode_limit": 140
        }

    def choose_sd_action(self, avail_action, robot_id):
        indices = np.where(np.array(avail_action[0:7]) == 1)[0]
        if indices.size == 0:
            return 7
        robot_type = self.robots.robots_type_id[robot_id]
        speed = self.robots.speed[robot_type]
        dis = [0. for i in range(len(indices))]
        for i, avail_act in enumerate(indices):
            task = self.task_window[avail_act]
            [_, _, request_id, task_type, destination_id, _] = task
            if task_type == 2 or task_type == 5:
                first_place = destination_id
            else:
                first_place = request_id
            dis[i] = distance(self.robots.robot_pos[robot_id], self.sites.sites_pos[first_place], speed)
        min_index = np.argmin(dis)
        # print("robot: ", robot_id, "avail_action: ", avail_action, "distance:", dis, "sd_choose_action: ", indices[min_index])
        return indices[min_index]

    def set_sd_episode_time_wait(self, sd_episode_time_wait):
        self.baseline = sd_episode_time_wait

    def save_task_assignments_to_file(self):
        with open('record/robot_task_assignments.txt', 'w') as f:
            for robot_id, assignments in enumerate(self.robot_task_assignments):
                f.write(f"Robot {robot_id} Task Assignments:\n")
                for assignment in assignments:
                    try:
                        f.write(f"\tTask Index: {assignment['task_index']:>4}  "
                                f"Requests Time: {assignment['requests_time']:>6}  "
                                f"Allocate Time: {assignment['allocate_time']:>7.1f}  "
                                f"Wait Time: {assignment['wait_time']:>6.1f}  "
                                f"Site ID: {assignment['site_id']:>3}  "
                                f"Task ID: {assignment['task_id']:>3}  "
                                f"Destination ID: {assignment['destination_id']:>4}  "
                                f"Service Time: {assignment['service_time']:>6}  "
                                f"Time on Road: {assignment['time_on_road']:>7.2f}  "
                                f"Total Time: {assignment['total_time']:>7.2f}  "
                                f"Reward: {assignment['reward']:>7.4f}\n")
                    except KeyError as e:
                        print(f"Error accessing key in assignment: {e}")


