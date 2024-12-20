from task_generator import generate_tasks
import pickle


def generate_fixed_tasks():
    all_tasks = []
    # 生成4个不同的任务列表
    for _ in range(4):
        tasks = generate_tasks()
        all_tasks.append(tasks)
    # 存储任务列表到文件
    with open('fixed_tasks.pkl', 'wb') as f:
        pickle.dump(all_tasks, f)
    return all_tasks


def generate_test_tasks():
    all_tasks = []
    # 生成100个不同的任务列表进行算法性能测试
    for _ in range(100):
        tasks = generate_tasks()
        all_tasks.append(tasks)
    # 存储六个任务列表到文件
    with open('test_tasks.pkl', 'wb') as f:
        pickle.dump(all_tasks, f)
    return all_tasks


def load_tasks_from_file(file_path):
    try:
        # 从文件中加载任务列表
        with open(file_path, 'rb') as f:
            tasks = pickle.load(f)
        return tasks
    except FileNotFoundError:
        print(f"文件 '{file_path}' 未找到.")
        return None
    except Exception as e:
        print(f"加载任务列表时出现错误: {e}")
        return None


if __name__ == '__main__':
    # generate_test_tasks()
    # generate_fixed_tasks()
    test_task = load_tasks_from_file('test_tasks.pkl')
    fixed_task = load_tasks_from_file('fixed_tasks.pkl')
    print(test_task[0])
    print(fixed_task[0])


