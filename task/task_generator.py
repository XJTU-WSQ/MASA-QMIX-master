"""
注：！！！目前缺少对于紧急事件的处理(老人摔倒，突发疾病，突发事故（火灾）---确认老人安全)
e.g.机器人B在 A点发现老人摔倒/突发疾病 向医护值班室C通过通信报送老人位置，等待医护人员到达后，协助医护人员运送老人。

任务：[时间，任务请求点，任务类型，任务目标点，服务时长]

任务类型(服从多项式分布)： task_types = ["紧急事件", "移动辅助任务", "送餐", "私人物品递送", "情感陪护", "康复训练"]

不同类型的任务，机器人前往执行任务的的次序不同：
case 1  "移动辅助任务"    和    "康复训练"         ：先去任务请求点，再前往目标点
case 2  "紧急事件"     和    "情感陪护"          ：直接抵达老人位置服务
case 3  "送餐"       和   "私人物品配送"        ：先去任务目标点，再前往任务请求点
"""
import numpy as np
import random
import unicodedata

TASK_INFO = ["Emergency", "Mobility Assistance Task", "Meal Delivery", "Personal Item Delivery", "Emotional Accompaniment", "Rehabilitation Training"]

LOCATIONS = ["Room1", "Room2", "Room3", "Room4", "Room5", "Room6", "Room7", "Room8", "Room9",
             "Room10", "Room11", "Room12", "Room13", "Room14", "Room15", "Room16", "Room17", "Room18",
             "Dining Room", "South Activity Area", "North Activity Area", "Activity Room",
             "Bathroom", "Duty Room", "Northwest Toilet", "Southeast Toilet"]

TASK_TYPES = {
    "general": [1, 2, 3, 4, 5],
    "canteen": [1, 4, 5],
    "toilet":  [1, 5]
}

TASK_PROBABILITIES = {
    "general": [0.6, 0.1, 0.1, 0.1, 0.1],
    "canteen": [0.6, 0.2, 0.2],
    "toilet":  [0.9, 0.1]
}


def generate_destination(task_type, site):
    all_sites = list(range(26))  # 共有26个位置
    if task_type == 0 or task_type == 4:
        destination_site = site
    elif task_type == 2:
        destination_site = 18
    else:
        available_sites = [s for s in all_sites if s != site]
        destination_site = np.random.choice(available_sites)
    return destination_site


def generate_task(site, time):
    task_info = [0, 0, 0, 0, 0]
    task_info[0] = time
    task_info[1] = site
    if site == 18:
        task_type = random.choices(TASK_TYPES["canteen"], TASK_PROBABILITIES["canteen"])[0]
    elif site == 24 or site == 25:
        task_type = random.choices(TASK_TYPES["toilet"], TASK_PROBABILITIES["toilet"])[0]
    else:
        task_type = random.choices(TASK_TYPES["general"], TASK_PROBABILITIES["general"])[0]
    destination = generate_destination(task_type, site)
    # 任务请求
    task_info[2] = task_type
    task_info[3] = destination
    service_time = 0
    # 根据任务类型不同，任务执行时间设计为一个范围内的随机数，且服从均匀分布 service_time = np.random.randint(60, 120)
    # task_types = ["紧急事件", "移动辅助任务", "送餐", "私人物品递送", "情感陪护", "康复训练"]
    if task_type == 0:
        service_time = 120
    elif task_type == 1:
        service_time = 30
    elif task_type == 2:
        service_time = 30
    elif task_type == 3:
        service_time = 30
    elif task_type == 4:
        service_time = 300
    elif task_type == 5:
        service_time = 300
    task_info[4] = service_time
    return task_info


def generate_uniform_process(time_window, n):
    """
    生成一个均匀分布的时间过程，确保任务在指定时间区间内均匀分布。
    :param time_window: 总的时间区间长度（单位秒），例如600秒
    :param n: 任务的数量
    :return: 返回任务的到达时间列表
    """
    return np.sort(np.random.uniform(0, time_window, n))


def generate_random_tasks():
    time_intervals = np.zeros((26, 10))
    tasks = []
    # 生成多个任务
    for site in range(len(LOCATIONS)):
        # 生成均匀分布的时间点而不是泊松分布
        time_intervals[site] = generate_uniform_process(3600, 10)  # 3600秒内生成10个任务请求
        for i in range(len(time_intervals[site])):
            time = time_intervals[site][i]
            task_item = generate_task(site, time)
            tasks.append(task_item)

    tasks = np.array(tasks, dtype=int)
    tasks = tasks[np.argsort(tasks[:, 0])]  # 按时间排序任务
    t_0 = tasks[0][0]
    for i in range(len(tasks)):
        tasks[i][0] -= t_0  # 使任务时间从0开始
    return tasks


def generate_tasks():
    all_tasks = generate_random_tasks()
    all_tasks = np.column_stack((np.arange(0, len(all_tasks)), all_tasks))
    return all_tasks


def get_display_width(text):
    """
    计算字符串的显示宽度（中文字符宽度为2，英文字符宽度为1）。
    """
    return sum(2 if unicodedata.east_asian_width(c) in "WFA" else 1 for c in text)


def pad_to_width(text, width):
    """
    按指定宽度填充字符串，使得显示宽度与指定宽度一致。
    """
    current_width = get_display_width(text)
    padding = width - current_width
    return text + " " * padding


if __name__ == "__main__":
    all_task = generate_tasks()

    # Define column names and initialize column widths
    columns = ["Task index", "Time", "Location", "Task Type", "Target", "Duration"]
    col_widths = [12, 12, 25, 30, 25, 12]  # 根据实际显示内容宽度调整

    # Open the file and write the table
    with open('tasks', 'w', encoding='utf-8') as f:
        # Write header
        header = "|".join(
            pad_to_width(columns[i], col_widths[i]) for i in range(len(columns))
        )
        f.write(header + "\n")
        f.write("-" * get_display_width(header) + "\n")

        # Write task data
        for task in all_task:
            row = [
                str(task[0]),  # Task index
                str(task[1]),  # Time
                str(LOCATIONS[task[2]]),  # Location
                str(TASK_INFO[task[3]]),  # Task Type
                str(LOCATIONS[task[4]]),  # Target
                str(task[5]),  # Duration
            ]
            line = "|".join(
                pad_to_width(row[i], col_widths[i]) for i in range(len(row))
            )
            f.write(line + "\n")


