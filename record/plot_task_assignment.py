import matplotlib.pyplot as plt


def read_task_assignments_from_file(path):
    robot_task_assignments = {}
    with open(path, 'r') as f:
        lines = f.readlines()
        i = 0
        while i < len(lines):
            if lines[i].startswith("Robot"):
                robot_id = int(lines[i].split()[1])
                assignments = []
                i += 1
                while i < len(lines) and not lines[i].startswith("Robot"):
                    assignment_info = lines[i].split(', ')
                    assignment = {
                        'task_index': int(assignment_info[0].split(': ')[1]),
                        'requests_time': int(assignment_info[1].split(': ')[1]),
                        'allocate_time': float(assignment_info[2].split(': ')[1]),
                        'wait_time': float(assignment_info[3].split(': ')[1]),
                        'site_id': int(assignment_info[4].split(': ')[1]),
                        'task_id': int(assignment_info[5].split(': ')[1]),
                        'destination_id': int(assignment_info[6].split(': ')[1]),
                        'service_time': int(assignment_info[7].split(': ')[1]),
                        'time_on_road': float(assignment_info[8].split(': ')[1]),
                        'total_time': float(assignment_info[9].split(': ')[1]),
                        'reward': float(assignment_info[10].split(': ')[1]),
                    }
                    assignments.append(assignment)
                    i += 1
                robot_task_assignments[robot_id] = assignments
            else:
                i += 1
    return robot_task_assignments


def plot_task_assignments(task_assignments):
    fig, ax = plt.subplots(figsize=(15, 8))

    for robot_id, assignments in task_assignments.items():
        allocate_times = [assignment['allocate_time'] for assignment in assignments]
        task_indices = [assignment['task_index'] for assignment in assignments]
        total_times = [assignment['total_time'] for assignment in assignments]

        ax.barh([robot_id] * len(allocate_times), total_times, left=allocate_times,
                label=f'Robot {robot_id}', alpha=0.7)

    ax.set_xlabel('Time')
    ax.set_ylabel('Robot-ID')

    # 设置纵轴刻度标签
    robot_ids = list(task_assignments.keys())
    ax.set_yticks(robot_ids)
    ax.set_yticklabels([f'robot{i}' for i in robot_ids])

    # 将图例移动到右上角
    ax.legend(bbox_to_anchor=(1, 1))

    plt.title('Task Allocation at Execution Time (total_time)')
    plt.show()


if __name__ == "__main__":
    file_path = 'robot_task_assignments.txt'  # 请替换为你保存的文件路径
    task_assignments = read_task_assignments_from_file(file_path)
    plot_task_assignments(task_assignments)
