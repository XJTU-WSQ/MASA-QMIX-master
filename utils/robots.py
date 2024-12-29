"""
养老服务机器人类：机器人的种类，技能和位置信息
robot类属性：
位置信息、类型 robotID、空闲的标识符isIdle、0-1技能集合
"""
from utils.util import *
from utils.sites import Sites


def cal_pos(start_pos, target_pos, speed, time_span):
    renew_pos = list(start_pos)  # 创建新列表以防止修改原始列表
    x_d = abs(start_pos[0] - target_pos[0])
    y_d = abs(start_pos[1] - target_pos[1])
    d = time_span * speed

    if d <= x_d:
        renew_pos[0] += (d if start_pos[0] < target_pos[0] else -d)
    elif x_d < d <= x_d + y_d:
        res = d - x_d
        renew_pos[0] = target_pos[0]
        renew_pos[1] += (res if start_pos[1] < target_pos[1] else -res)
    else:
        renew_pos = target_pos
    return tuple(renew_pos)  # 将结果转换为元组返回


class Robots:

    def __init__(self):
        # 机器人数
        self.sites = Sites()
        self.n_wheelchair = 8
        self.n_delivery = 2
        self.n_private_delivery = 2
        self.n_company = 3
        self.n_walking = 3
        # 技能数
        self.num_skills = 5
        # 机器人名称
        self.robot_info = ["智能轮椅机器人", "开放式递送机器人", "箱式递送机器人", "情感陪护机器人", "辅助行走机器人"]
        # 机器人速度
        self.speed = [1.5, 1.0, 1.0, 1.0, 1.0]
        # 机器人总数
        self.num_robots = 18
        # 机器人的类型
        self.robots_type_id = [0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 2, 2, 3, 3, 3, 4, 4, 4]
        # 机器人的技能信息
        self.robots_skills = np.zeros([self.num_robots, self.num_skills])
        # robot_sites_pos 分别对应①-⑤类型机器人的停靠点
        self.robot_sites_pos = np.array([
            [1, 35], [1, 34], [1, 33], [1, 32], [1, 31], [1, 30], [1, 29], [1, 28],
            [1, 27], [1, 26],
            [1, 25], [1, 24],
            [1, 23], [1, 22], [1, 21],
            [1, 19], [1, 18], [1, 17]
        ])
        # [time_on_road_1, service_time, time_on_road_2]
        self.robots_time = np.zeros([self.num_robots, 3])
        # 机器人位置初始化
        self.robot_pos = self.robot_sites_pos
        # 机器人执行任务的信息 robots_tasks_info[i] = [task_index, requests_time, site_id, task_id, destination_id, service_time]
        self.robots_tasks_info = np.zeros([self.num_robots, 6], dtype=int)
        # 获取每个机器人所掌握的技能集
        for i in range(self.num_robots):
            self.robots_skills[i] = self.get_skills(i)

    # 机器人能力：①辅助老人移动能力 ② 送餐能力 ③递送能力 ④ 陪伴能力 ⑤康复训练能力
    def get_skills(self, agent_id):
        robot_type_id = self.robots_type_id[agent_id]
        if robot_type_id == 0:
            return [1, 0, 0, 0, 0]   # 类型 0：智能轮椅机器人
        if robot_type_id == 1:
            return [0, 1, 0, 0, 0]   # 类型 1：开放式递送机器人
        if robot_type_id == 2:
            return [0, 0, 1, 0, 0]   # 类型 2：箱式递送机器人
        if robot_type_id == 3:
            return [0, 0, 0, 1, 0]   # 类型 3：情感陪护机器人
        if robot_type_id == 4:
            return [0, 0, 0, 0, 1]   # 类型 4：辅助行走机器人

    # 执行任务（暂时没有考虑任务类型为1时，紧急情况的处理）
    # 输入的范围：task_id:0-6, robot_id:0-14, site_id:0-25, destination_id:0-25
    # 返回：机器人执行完全部任务所需时长
    def execute_task(self, robot_id, task):
        # 机器人的任务信息：
        [task_index, requests_time, site_id, task_id, destination_id, service_time] = task
        self.robots_tasks_info[robot_id] = [task_index, requests_time, site_id, task_id, destination_id, service_time]
        # 机器人的类型和速度
        robot_type_id = self.robots_type_id[robot_id]
        speed = self.speed[robot_type_id]
        # task_types = ["紧急事件", "移动辅助任务", "送餐", "私人物品递送", "情感陪护", "康复训练"]
        if task_id == 2 or task_id == 3:  # 送餐/私人物品配送：先去目标点，再去任务请求点
            time_on_road_1 = count_path_on_road(self.robot_pos[robot_id], self.sites.sites_pos[destination_id],
                                                speed)
            time_on_road_2 = count_path_on_road(self.sites.sites_pos[destination_id], self.sites.sites_pos[site_id],
                                                speed)
        else:  # 其他：先去任务请求点，再去目标点
            time_on_road_1 = count_path_on_road(self.robot_pos[robot_id], self.sites.sites_pos[site_id], speed)
            time_on_road_2 = count_path_on_road(self.sites.sites_pos[site_id], self.sites.sites_pos[destination_id],
                                                speed)
        # 保留一个执行任务时，在路上的时间，在这个过程中如果突发紧急情况，可以对机器人任务进行重置，优先执行紧急任务。
        self.robots_time[robot_id] = [time_on_road_1, service_time, time_on_road_2]
        total_time = time_on_road_1 + service_time + time_on_road_2
        return time_on_road_1, total_time

    # 更新机器人位置信息
    # 输入：上一个step的多智能体的联合动作，选择去哪个任务请求点，同时要有上一个step的任务列表和任务目标点列表。上一个step花费的时间。
    # 代码中需要获得机器人的ID，机器人选取的任务，机器人的目标点，设计一个机器人移动路径，并根据step移动的时间更新机器人的位置信息。
    def renew_position(self, robot_id, task_id, site_id, destination_id, time_span):
        time_table = self.robots_time[robot_id]
        time_phase1, time_phase2, time_phase3 = time_table
        robot_type_id = self.robots_type_id[robot_id]
        speed = self.speed[robot_type_id]

        if task_id == 2 or task_id == 3:
            site_pos, destination_pos = self.sites.sites_pos[destination_id], self.sites.sites_pos[site_id]
        else:
            site_pos, destination_pos = self.sites.sites_pos[site_id], self.sites.sites_pos[destination_id]

        if time_span < time_phase1:
            t_1 = time_span
            self.robot_pos[robot_id] = cal_pos(self.robot_pos[robot_id], site_pos, speed, t_1)
            self.robots_time[robot_id][0] = time_phase1 - time_span
            return 0

        elif time_phase1 <= time_span < time_phase1 + time_phase2:
            self.robot_pos[robot_id] = site_pos
            self.robots_time[robot_id][0] = 0
            self.robots_time[robot_id][1] = time_phase1 + time_phase2 - time_span
            return 0

        elif time_phase1 + time_phase2 <= time_span < time_phase1 + time_phase2 + time_phase3:
            t_3 = time_span - (time_phase1 + time_phase2)
            if time_phase1 > 0 or time_phase2 > 0:
                self.robot_pos[robot_id] = site_pos
            self.robot_pos[robot_id] = cal_pos(self.robot_pos[robot_id], destination_pos, speed, t_3)
            self.robots_time[robot_id][0] = 0
            self.robots_time[robot_id][1] = 0
            self.robots_time[robot_id][2] = time_phase3 - t_3
            return 0
        else:
            self.robot_pos[robot_id] = destination_pos
            self.robots_time[robot_id] = [0, 0, 0]
            return 1
