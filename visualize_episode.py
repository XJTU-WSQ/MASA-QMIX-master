import json


def visualize_episode(filepath):
    """
    可视化每个 step 的详细信息，包括任务、机器人、观测和动作等。
    :param filepath: 存储 episode 数据的 JSON 文件路径。
    """
    with open(filepath, 'r', encoding='utf-8') as f:
        episode_data = json.load(f)

    for step_info in episode_data:
        print(f"Step: {step_info.get('step')}")
        print(f"Time: {step_info.get('time')}")

        # 显示当前待分配任务信息
        print(f"Number of Waiting Tasks: {step_info.get('num_waiting_tasks', 0)}")
        print("Waiting Tasks:")
        for task in step_info.get("waiting_tasks", []):
            print(f"  - Task Index: {task.get('task_index')}")
            print(f"    Request Position: {task.get('request_pos')}")
            print(f"    Target Position: {task.get('target_pos')}")
            print(f"    Priority: {task.get('priority')}")
            print(f"    Service Time: {task.get('service_time')}")
            print(f"    Wait Time: {task.get('wait_time')}")

        # 显示机器人信息
        print("Robots Info:")
        for robot in step_info.get("robots_info", []):
            print(f"  - Robot ID: {robot.get('robot_id')}")
            print(f"    Position: {robot.get('position')}")
            print(f"    Status: {robot.get('status')}")
            if robot.get("status") == "busy":
                print(f"    Current Task: {robot.get('current_task')}")

        # 显示动作和奖励信息
        print("Global Info:")
        print(f"  State: {step_info.get('state')}")
        print(f"  Observations: {step_info.get('observations')}")
        print(f"  Actions Taken: {step_info.get('actions')}")
        print(f"  Step Reward: {step_info.get('reward')}")

        print("-" * 80)


if __name__ == "__main__":
    # 修改文件路径以适配具体的 JSON 文件
    input_file = "logs/episode_2_0.json"  # 替换为实际的 JSON 文件路径
    visualize_episode(input_file)
