import numpy as np
import os
from common.rollout import RolloutWorker
from agent.agent import Agents
from common.replay_buffer import ReplayBuffer
import sys
from test_task import load_tasks_from_file
from RD_rules import random_agent_wrapper
from SD_rules import sd_rules_agent_wrapper


class Runner:
    def __init__(self, env, args):
        self.env = env
        self.agents = Agents(args)
        self.rolloutWorker = RolloutWorker(env, self.agents, args)
        self.buffer = ReplayBuffer(args)
        self.args = args

        self.epoch_idx = []
        self.rewards = []
        self.epoch_time_on_roads = []
        self.epoch_time_wait = []
        self.evaluate_index = 0
        self.episode_index = 0

        self.save_path = self.args.result_dir + '/' + args.alg
        if not os.path.exists(self.save_path):
            os.makedirs(self.save_path)
        self.evaluate_tasks = load_tasks_from_file('task/fixed_tasks.pkl')
        # evaluate 的返回值
        self.qmix_reward = [[] for i in range(len(self.evaluate_tasks))]
        self.qmix_time_wait = [[] for i in range(len(self.evaluate_tasks))]
        self.sd_reward = [[] for i in range(len(self.evaluate_tasks))]
        self.sd_time_wait = [[] for i in range(len(self.evaluate_tasks))]
        self.rd_reward = [[] for i in range(len(self.evaluate_tasks))]
        self.rd_time_wait = [[] for i in range(len(self.evaluate_tasks))]
        self.sd_episode_reward = [0 for i in range(len(self.evaluate_tasks))]
        self.sd_episode_time_wait = [0 for i in range(len(self.evaluate_tasks))]
        for tasks_idx, task in enumerate(self.evaluate_tasks):
            self.sd_episode_reward[tasks_idx], sd_episode_time_on_road, self.sd_episode_time_wait[tasks_idx], _ = sd_rules_agent_wrapper(task)

    def run(self):
        train_steps = 0
        for epoch in range(self.args.n_epoch):
            # 显示输出
            if epoch % self.args.evaluate_cycle == 0 and epoch != 0:
                self.evaluate()
            episodes = []
            r_s = []
            t1_s = []
            t2_s = []
            # 收集self.args.n_episodes个episodes

            for episode_idx in range(self.args.n_episodes):
                episode, episode_reward, episode_time_on_road, episode_time_wait, terminated = self.rolloutWorker.generate_episode()
                text = 'train episode:  {}, rewards:  {:.2f}, time_on_road:  {:.2f}, time_wait:  {:.2f}, done: {:}\n'
                sys.stdout.write(
                    text.format(self.episode_index, episode_reward, episode_time_on_road, episode_time_wait,
                                terminated))
                self.episode_index += 1
                episodes.append(episode)
                r_s.append(episode_reward)
                t1_s.append(episode_time_on_road)
                t2_s.append(episode_time_wait)

            epoch_average_reward = sum(r_s) / len(r_s)
            epoch_average_time_wait = sum(t2_s) / len(t2_s)
            self.epoch_idx.append(epoch)
            if len(self.rewards) == 0:
                reward = epoch_average_reward
            else:
                reward = 0.1*epoch_average_reward + 0.9*sum(self.rewards) / len(self.rewards)

            if len(self.epoch_time_wait) == 0:
                average_time_wait = epoch_average_time_wait
            else:
                average_time_wait = 0.1*epoch_average_time_wait + 0.9*sum(self.epoch_time_wait) / len(self.epoch_time_wait)
            print("train_epoch:", epoch, "average_reward:", reward, "average_time_wait:", average_time_wait)
            self.rewards.append(reward)
            self.epoch_time_on_roads.append(sum(t1_s) / self.args.n_episodes)
            self.epoch_time_wait.append(average_time_wait)

            if epoch % 100 == 0:
                data_r = list(zip(self.epoch_idx, self.rewards))
                data_t2 = list(zip(self.epoch_idx, self.epoch_time_wait))
                np.savetxt('data/rewards_data.csv', data_r, delimiter=',')
                np.savetxt('data/time_wait.csv', data_t2, delimiter=',')

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

    def evaluate(self):
        for tasks_idx, task in enumerate(self.evaluate_tasks):
            # print('tasks_idx:', tasks_idx, 'task:', task)
            qmix_episode_reward, qmix_episode_time_on_road, qmix_episode_time_wait, _ = self.rolloutWorker.evaluate_episode(task)
            self.qmix_reward[tasks_idx].append(qmix_episode_reward)
            self.qmix_time_wait[tasks_idx].append(qmix_episode_time_wait)
            # sd 奖励函数这里需要修改
            self.sd_reward[tasks_idx].append(0)
            self.sd_time_wait[tasks_idx].append(self.sd_episode_time_wait[tasks_idx])
            rd_episode_reward, rd_episode_time_on_road, rd_episode_time_wait, _ = random_agent_wrapper(task)
            self.rd_reward[tasks_idx].append(rd_episode_reward)
            self.rd_time_wait[tasks_idx].append(rd_episode_time_wait)

        # 保存数据
        save_data('qmix', self.qmix_reward, self.qmix_time_wait)
        save_data('sd', self.sd_reward, self.sd_time_wait)
        save_data('rd', self.rd_reward, self.rd_time_wait)

        print("-------------------------------------")
        text = 'qmix_evaluate_index:  {}, qmix_ave_rewards:  {:.2f}, qmix_ave_time_wait:  {:.2f}\n'
        sys.stdout.write(text.format(self.evaluate_index,
                                     sum(row[self.evaluate_index] for row in self.qmix_reward) / len(self.qmix_reward),
                                     sum(row[self.evaluate_index] for row in self.qmix_time_wait) / len(
                                         self.qmix_time_wait)))
        print("-------------------------------------")
        text_1 = 'sd_evaluate_index:  {}, sd_ave_rewards:  {:.2f}, sd_ave_time_wait:  {:.2f}\n'
        sys.stdout.write(text_1.format(self.evaluate_index,
                                       sum(row[self.evaluate_index] for row in self.sd_reward) / len(self.sd_reward),
                                       sum(row[self.evaluate_index] for row in self.sd_time_wait) / len(
                                         self.sd_time_wait)))
        print("-------------------------------------")
        text_2 = 'rd_evaluate_index:  {}, rd_ave_rewards:  {:.2f}, rd_ave_time_wait:  {:.2f}\n'
        sys.stdout.write(text_2.format(self.evaluate_index,
                                       sum(row[self.evaluate_index] for row in self.rd_reward) / len(self.rd_reward),
                                       sum(row[self.evaluate_index] for row in self.rd_time_wait) / len(
                                         self.rd_time_wait)))

        self.evaluate_index += 1
        print("-------------------------------------")


def save_data(prefix, reward, time_wait):
    data_r = list(reward)
    data_t2 = list(time_wait)

    np.savetxt(f'data/{prefix}_episode_rewards.csv', data_r, delimiter=',')
    np.savetxt(f'data/{prefix}_episode_time_wait.csv', data_t2, delimiter=',')



