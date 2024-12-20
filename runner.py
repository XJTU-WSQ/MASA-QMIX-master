import json
import numpy as np
import os
from common.rollout import RolloutWorker
from agent.agent import Agents
from common.replay_buffer import ReplayBuffer
import sys
from test_task import load_tasks_from_file


class Runner:
    def __init__(self, env, args):
        self.env = env
        self.agents = Agents(args)
        self.rolloutWorker = RolloutWorker(env, self.agents, args)
        self.buffer = ReplayBuffer(args)
        self.args = args
        self.rewards = []
        self.evaluate_index = 0
        self.episode_index = 0
        self.save_path = self.args.result_dir + '/' + args.alg
        if not os.path.exists(self.save_path):
            os.makedirs(self.save_path)
        self.evaluate_tasks = load_tasks_from_file('task/fixed_tasks.pkl')
        # evaluate 的返回值
        self.qmix_reward = [[] for i in range(len(self.evaluate_tasks))]

    def run(self):
        train_steps = 0
        metrics = {
            "epoch_rewards": [],
            "completed_tasks": [],
            "conflict_rates": [],
            "resolved_conflict_rates": [],
            "avg_wait_times": []
        }

        for epoch in range(self.args.n_epoch):
            # 显示输出
            if epoch % self.args.evaluate_cycle == 0 and epoch != 0:
                self.evaluate()
            episodes = []
            # 收集self.args.n_episodes个episodes
            epoch_metrics = {"rewards": [], "completed_tasks": 0, "conflicts": 0, "resolved_conflicts": 0, "avg_wait_time": 0}

            for episode_idx in range(self.args.n_episodes):
                episode, episode_reward, terminated, episode_data, episode_metrics = self.rolloutWorker.generate_episode()
                filename = f"logs/episode_{epoch}_{episode_idx}.json"
                with open(filename, 'w') as f:
                    json.dump(episode_data, f, indent=4)
                text = 'train episode:  {}, rewards:  {:.2f},  done: {:}\n'
                sys.stdout.write(
                    text.format(self.episode_index, episode_reward, terminated))
                self.episode_index += 1
                episodes.append(episode)
                epoch_metrics["rewards"].append(episode_reward)
                epoch_metrics["completed_tasks"] += episode_metrics["completed_tasks"]
                epoch_metrics["conflicts"] += episode_metrics["conflicted_tasks"]
                epoch_metrics["resolved_conflicts"] += episode_metrics["resolved_conflicts"]
                epoch_metrics["avg_wait_time"] += episode_metrics["average_wait_time"]

            # 记录每个 epoch 的统计量
            metrics["epoch_rewards"].append(sum(epoch_metrics["rewards"]) / len(epoch_metrics["rewards"]))
            metrics["completed_tasks"].append(epoch_metrics["completed_tasks"] / self.args.n_episodes)
            metrics["conflict_rates"].append(epoch_metrics["conflicts"] / self.args.n_episodes)
            metrics["resolved_conflict_rates"].append(epoch_metrics["resolved_conflicts"] / self.args.n_episodes)
            metrics["avg_wait_times"].append(epoch_metrics["avg_wait_time"] / self.args.n_episodes)

            print(f"Epoch {epoch}: Avg Reward={metrics['epoch_rewards'][-1]:.2f}, "
                  f"Task Completion={metrics['completed_tasks'][-1]:.2f}, "
                  f"Conflict Rate={metrics['conflict_rates'][-1]:.2f}, "
                  f"Resolved Conflict Rate={metrics['resolved_conflict_rates'][-1]:.2f}, "
                  f"Avg Wait Time={metrics['avg_wait_times'][-1]:.2f}")

            # episode的每一项都是一个(1, episode_len, n_agents, 具体维度)四维数组，下面要把所有episode的obs拼在一起
            episode_batch = episodes[0]
            episodes.pop(0)
            for episode in episodes:
                for key in episode_batch.keys():
                    episode_batch[key] = np.concatenate((episode_batch[key], episode[key]), axis=0)
            self.buffer.store_episode(episode_batch)
            for train_step in range(self.args.train_steps):
                mini_batch = self.buffer.sample(min(self.buffer.current_size, self.args.batch_size))
                self.agents.train(mini_batch, train_steps)
                train_steps += 1

        with open("train_metrics.json", "w") as f:
            json.dump(metrics, f)

    def evaluate(self):
        eval_metrics = {
            "evaluation_rewards": [],
            "completed_tasks": 0,
            "conflicts": 0,
            "resolved_conflicts": 0,
            "avg_wait_time": 0
        }

        for tasks_idx, task in enumerate(self.evaluate_tasks):
            qmix_episode_reward, terminated, episode_metrics = self.rolloutWorker.evaluate_episode(task)
            eval_metrics["evaluation_rewards"].append(qmix_episode_reward)
            eval_metrics["completed_tasks"] += episode_metrics["completed_tasks"]
            eval_metrics["conflicts"] += episode_metrics["conflicted_tasks"]
            eval_metrics["resolved_conflicts"] += episode_metrics["resolved_conflicts"]
            eval_metrics["avg_wait_time"] += episode_metrics["average_wait_time"]

        # 计算平均值
        num_evaluate_tasks = len(self.evaluate_tasks)
        eval_metrics["avg_reward"] = sum(eval_metrics["evaluation_rewards"]) / num_evaluate_tasks
        eval_metrics["avg_completed_tasks"] = eval_metrics["completed_tasks"] / num_evaluate_tasks
        eval_metrics["avg_conflict_rate"] = eval_metrics["conflicts"] / num_evaluate_tasks
        eval_metrics["avg_resolved_conflict_rate"] = eval_metrics["resolved_conflicts"] / num_evaluate_tasks
        eval_metrics["avg_wait_time"] = eval_metrics["avg_wait_time"] / num_evaluate_tasks

        # 打印评估结果
        print(f"Evaluation Results: "
              f"Avg Reward={eval_metrics['avg_reward']:.2f}, "
              f"Task Completion Rate={eval_metrics['avg_completed_tasks']:.2f}, "
              f"Conflict Rate={eval_metrics['avg_conflict_rate']:.2f}, "
              f"Resolved Conflict Rate={eval_metrics['avg_resolved_conflict_rate']:.2f}, "
              f"Avg Wait Time={eval_metrics['avg_wait_time']:.2f}")

        # 保存评估统计量
        with open("eval_metrics.json", "w") as f:
            json.dump(eval_metrics, f)





