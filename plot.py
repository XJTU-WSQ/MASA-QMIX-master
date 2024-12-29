import json
import matplotlib.pyplot as plt


def plot_episode_data(filename):
    """
    读取 JSON 文件并可视化每个 step 的信息，包括全局 state。
    """
    # 读取 JSON 文件
    with open(filename, 'r') as f:
        data = json.load(f)

    # 遍历每个 step 的数据
    for step_data in data:
        step = step_data["step"]
        print(f"===== Step {step} =====")

        # 打印全局信息
        print("Global Information:")
        print(f"  Total Reward: {step_data['reward']}")
        print(f"  Task Rewards: {step_data['task_rewards']}")
        print(f"  Conflict Penalty: {step_data['conflict_penalty']}")
        print(f"  Overdue Penalty: {step_data['overdue_penalty']}")
        print(f"  Service Cost Penalty: {step_data['total_service_cost_penalty']}")
        print(f"  Conflict Count: {step_data['conflict_count']}\n")

        # 打印全局状态 state
        print("Global State (state):")
        print(step_data["state"])
        print()

        # 打印机器人信息
        print("Robot Information:")
        for robot_id, robot_state in enumerate(step_data["robots_state"]):
            print(f"  Robot {robot_id}: State = {robot_state}, Action = {step_data['actions'][robot_id]}")

        print("\nAvailable Actions:")
        for robot_id, avail_action in enumerate(step_data["avail_actions"]):
            print(f"  Robot {robot_id}: {avail_action}")

        # 打印任务窗口信息
        print("\nTask Window Information:")
        for task in step_data["task_window"]:
            print(f"  Task {task[0]} - Start Time: {task[1]}, Location: {task[2]}, Type: {task[3]}, Target: {task[4]}, Duration: {task[5]}")

        print("\nObservations:")
        for robot_id, obs in enumerate(step_data["obs"]):
            print(f"  Robot {robot_id}: Observation = {obs}")

        # 可视化机器人状态
        plt.figure(figsize=(12, 6))
        plt.bar(range(len(step_data["robots_state"])), step_data["robots_state"], color='blue', alpha=0.7, label="Robot State (0: Idle, 1: Busy)")
        plt.xticks(range(len(step_data["robots_state"])))
        plt.xlabel("Robot ID")
        plt.ylabel("State")
        plt.title(f"Robot State and Actions at Step {step}")
        plt.legend()
        plt.grid()

        for i, action in enumerate(step_data["actions"]):
            plt.text(i, step_data["robots_state"][i] + 0.1, f"A{action}", ha='center', fontsize=8)

        plt.show()

    # 全局奖励和冲突统计
    steps = [step["step"] for step in data]
    rewards = [step["reward"] for step in data]
    conflict_counts = [step["conflict_count"] for step in data]

    # 奖励趋势
    plt.figure(figsize=(10, 6))
    plt.plot(steps, rewards, label="Total Reward", marker='o')
    plt.xlabel("Steps")
    plt.ylabel("Reward")
    plt.title("Total Reward over Steps")
    plt.legend()
    plt.grid()
    plt.show()

    # 冲突数量
    plt.figure(figsize=(10, 6))
    plt.bar(steps, conflict_counts, color='orange', alpha=0.7, label="Conflict Count")
    plt.xlabel("Steps")
    plt.ylabel("Conflict Count")
    plt.title("Conflict Count over Steps")
    plt.legend()
    plt.grid()
    plt.show()


# 调用函数并传入 JSON 文件路径
plot_episode_data("./episode_logs/episode_0.json")
