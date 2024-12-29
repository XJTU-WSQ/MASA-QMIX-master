import numpy as np
import os
from torch.utils.tensorboard import SummaryWriter
from common.rollout import RolloutWorker
from agent.agent import Agents
from common.replay_buffer import ReplayBuffer
import sys
from policy.qmix import QMIX
from datetime import datetime


class Runner:
    def __init__(self, env, args):
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")  # 格式化为常见的时间格式
        self.writer = SummaryWriter(log_dir=f"runs/{args.run_name}_{timestamp}")
        self.qmix = QMIX(args, writer=self.writer)
        self.episode_count = 0
        self.test_episode_count = 0
        self.env = env
        self.agents = Agents(args, writer=self.writer)
        self.rolloutWorker = RolloutWorker(env, self.agents, args)
        self.buffer = ReplayBuffer(args)
        self.args = args
        self.episode_rewards = []
        self.save_path = self.args.result_dir + '/' + args.alg
        if not os.path.exists(self.save_path):
            os.makedirs(self.save_path)

    def run(self):
        train_steps = 0
        for epoch in range(self.args.n_epoch):
            # 显示输出
            if epoch % self.args.evaluate_cycle == 0:  # and epoch != 0:
                episode_reward = self.evaluate()
                print("\nevaluate_average_reward:", episode_reward, "epoch:", epoch)
            episode, episode_reward, terminated, stats = self.rolloutWorker.generate_episode(epoch)
            # 写入 TensorBoard
            self.writer.add_scalar("Train/Episode Reward", stats["episode_reward"], self.episode_count)
            self.writer.add_scalar("Train/Conflicts", stats["conflicts"], self.episode_count)
            self.writer.add_scalar("Train/Wait Time", stats["wait_time"], self.episode_count)
            self.writer.add_scalar("Train/Completed Tasks", stats["completed_tasks"], self.episode_count)
            self.writer.add_scalar("Train/Task Completion Rate", stats["completion_rate"], self.episode_count)
            self.writer.add_scalar("Train/Task Rewards", stats["task_rewards"], self.episode_count)
            self.writer.add_scalar("Train/Wait Penalty", stats["wait_penalty"], self.episode_count)
            self.writer.add_scalar("Train/Service Cost Penalty", stats["service_cost_penalty"], self.episode_count)
            self.episode_count += 1

            text = 'train_epoch:  {}, train_rewards:  {:.2f},  done: {:}\n'
            sys.stdout.write(text.format(epoch, episode_reward, terminated))
            # episode的每一项都是一个(1, episode_len, n_agents, 具体维度)四维数组，下面要把所有episode的obs拼在一起
            self.buffer.store_episode(episode)
            for train_step in range(self.args.train_steps):
                mini_batch = self.buffer.sample(min(self.buffer.current_size, self.args.batch_size))
                self.agents.train(mini_batch, train_steps)
                train_steps += 1
        self.writer.close()

    def evaluate(self):
        """
        测试阶段：记录测试的指标
        """
        total_test_reward = 0
        total_test_conflicts = 0
        total_test_wait_time = 0
        total_test_completed_tasks = 0

        total_test_task_rewards = 0
        total_test_wait_penalty = 0
        total_test_service_cost_penalty = 0
        total_test_completion_rate = 0
        num_test_episodes = self.args.evaluate_epoch

        for _ in range(num_test_episodes):
            _, test_reward, _, test_info = self.rolloutWorker.generate_episode(evaluate=True)

            total_test_reward += test_reward
            total_test_conflicts += test_info["conflicts"]
            total_test_wait_time += test_info["wait_time"]
            total_test_completed_tasks += test_info["completed_tasks"]
            total_test_task_rewards += test_info["task_rewards"]
            total_test_wait_penalty += test_info["wait_penalty"]
            total_test_service_cost_penalty += test_info["service_cost_penalty"]

            # 计算完成率
            total_test_completion_rate += test_info["completed_tasks"] / len(self.env.tasks_array)

        # 计算平均测试指标
        avg_test_reward = total_test_reward / num_test_episodes
        avg_test_conflicts = total_test_conflicts / num_test_episodes
        avg_test_wait_time = total_test_wait_time / num_test_episodes
        avg_test_completed_tasks = total_test_completed_tasks / num_test_episodes
        avg_test_task_rewards = total_test_task_rewards / num_test_episodes
        avg_test_wait_penalty = total_test_wait_penalty / num_test_episodes
        avg_test_service_cost_penalty = total_test_service_cost_penalty / num_test_episodes
        avg_test_completion_rate = total_test_completion_rate / num_test_episodes

        # 写入 TensorBoard
        self.writer.add_scalar("Test/Average Reward", avg_test_reward, self.test_episode_count)
        self.writer.add_scalar("Test/Average Conflicts", avg_test_conflicts, self.test_episode_count)
        self.writer.add_scalar("Test/Average Wait Time", avg_test_wait_time, self.test_episode_count)
        self.writer.add_scalar("Test/Average Completed Tasks", avg_test_completed_tasks, self.test_episode_count)
        self.writer.add_scalar("Test/Average Task Completion Rate", avg_test_completion_rate, self.test_episode_count)
        self.writer.add_scalar("Test/Average Task Rewards", avg_test_task_rewards, self.test_episode_count)
        self.writer.add_scalar("Test/Average Wait Penalty", avg_test_wait_penalty, self.test_episode_count)
        self.writer.add_scalar("Test/Average Service Cost Penalty", avg_test_service_cost_penalty, self.test_episode_count)

        self.test_episode_count += 1
        return avg_test_reward





