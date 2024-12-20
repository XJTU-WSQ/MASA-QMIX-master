import json
import csv


def save_detailed_episode_with_observations(filepath, output_file):
    """
    将 JSON 文件中的每个 step 数据展开为详细 CSV 表格，包括每个机器人的观测信息。
    :param filepath: JSON 文件路径。
    :param output_file: 输出 CSV 文件路径。
    """
    with open(filepath, 'r') as f:
        episode_data = json.load(f)

    rows = []
    for step_info in episode_data:
        step = step_info["step"]
        time = step_info["time"]
        step_reward = step_info["reward"]
        actions = step_info["actions"]
        global_observation = step_info["state"]
        observations = step_info.get("observations", [])

        # Add waiting tasks information
        for task in step_info["waiting_tasks"]:
            rows.append({
                "Step": step,
                "Time": time,
                "Type": "Task",
                "Task Index": task["task_index"],
                "Request Position": task["request_pos"],
                "Target Position": task["target_pos"],
                "Priority": task["priority"],
                "Service Time": task["service_time"],
                "Wait Time": task["wait_time"],
                "Robot ID": None,
                "Robot Position": None,
                "Robot Status": None,
                "Current Task": None,
                "Observation": None,
                "Global Observation": None,
                "Actions": None,
                "Step Reward": step_reward,
            })

        # Add robot information and their observations
        for i, robot in enumerate(step_info["robots_info"]):
            observation = observations[i] if i < len(observations) else None
            rows.append({
                "Step": step,
                "Time": time,
                "Type": "Robot",
                "Task Index": None,
                "Request Position": None,
                "Target Position": None,
                "Priority": None,
                "Service Time": None,
                "Wait Time": None,
                "Robot ID": robot["robot_id"],
                "Robot Position": robot["position"],
                "Robot Status": robot["status"],
                "Current Task": robot.get("current_task"),
                "Observation": observation,  # Assign corresponding observation
                "Global Observation": None,
                "Actions": None,
                "Step Reward": step_reward,
            })

        # Add global observation and actions
        rows.append({
            "Step": step,
            "Time": time,
            "Type": "Global",
            "Task Index": None,
            "Request Position": None,
            "Target Position": None,
            "Priority": None,
            "Service Time": None,
            "Wait Time": None,
            "Robot ID": None,
            "Robot Position": None,
            "Robot Status": None,
            "Current Task": None,
            "Observation": None,
            "Global Observation": global_observation,
            "Actions": actions,
            "Step Reward": step_reward,
        })

    # Define CSV headers
    headers = [
        "Step", "Time", "Type", "Task Index", "Request Position", "Target Position",
        "Priority", "Service Time", "Wait Time", "Robot ID", "Robot Position",
        "Robot Status", "Current Task", "Observation", "Global Observation", "Actions",
        "Step Reward"
    ]

    # Save to CSV using built-in csv module
    with open(output_file, 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=headers)
        writer.writeheader()
        writer.writerows(rows)

    print(f"Detailed episode data with observations saved to {output_file}")


# Paths for input JSON file and output CSV file
file_path = 'logs/episode_3_0.json'
output_csv_path = 'detailed_episode_with_observations.csv'

# Run the processing function
save_detailed_episode_with_observations(file_path, output_csv_path)
