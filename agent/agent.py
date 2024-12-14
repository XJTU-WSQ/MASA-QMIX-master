import numpy as np
import torch
from policy.qmix import QMIX


# 这里是否合适？
def random_choice_with_mask(avail_actions):
    temp = []    # 所有机器人都有不做分配这个动作
    for i, eve in enumerate(avail_actions):
        if eve == 1:
            temp.append(i)
    return np.random.choice(temp, 1, False)[0]


class Agents:
    def __init__(self, args):
        self.n_actions = args.n_actions
        self.n_agents = args.n_agents
        self.state_shape = args.state_shape
        self.obs_shape = args.obs_shape
        self.policy = QMIX(args)
        self.args = args
        print('Init Agents')

    def choose_action(self, obs, last_action, agent_num, avail_actions, epsilon):
        inputs = obs.copy()
        #  avail_actions_ind = np.nonzero(avail_actions)[0]  # index of actions which can be choose 存在不同
        # transform agent_num to one-hot vector
        agent_id = np.zeros(self.n_agents)
        agent_id[agent_num] = 1.
        # 需要思考上一个动作，是否会对当前时刻的任务分配产生影响？
        if self.args.last_action:
            inputs = np.hstack((inputs, last_action))
        if self.args.reuse_network:
            inputs = np.hstack((inputs, agent_id))
        hidden_state = self.policy.eval_hidden[:, agent_num, :]
        # transform the shape of inputs from (42,) to (1,42)
        inputs = torch.tensor(inputs, dtype=torch.float32).unsqueeze(0)  # [[]]  -> []
        avail_actions = torch.tensor(avail_actions, dtype=torch.float32).unsqueeze(0)
        if self.args.cuda:
            inputs = inputs.cuda()
            hidden_state = hidden_state.cuda()
        # get q value
        q_value, self.policy.eval_hidden[:, agent_num, :] = self.policy.eval_rnn(inputs, hidden_state)
        # choose action from q value
        q_value[avail_actions == 0.0] = -float("inf")
        if np.random.uniform() < epsilon:
            action = random_choice_with_mask(avail_actions[0])
        else:
            action = torch.argmax(q_value).cpu()
        return action

    def _get_max_episode_len(self, batch):
        terminated = batch['terminated']
        episode_num = terminated.shape[0]
        max_episode_len = 0
        # 由于episodelimit的长度内没有terminal==1，所以导致max_episode_len == 0
        for episode_idx in range(episode_num):
            for transition_idx in range(self.args.episode_limit):
                if terminated[episode_idx, transition_idx, 0] == 1:
                    if transition_idx + 1 >= max_episode_len:
                        max_episode_len = transition_idx + 1
                    break
                if transition_idx == self.args.episode_limit - 1:
                    max_episode_len = self.args.episode_limit
        return max_episode_len

    def train(self, batch, train_step, epsilon=None):

        # different episode has different length, so we need to get max length of the batch
        max_episode_len = self._get_max_episode_len(batch)
        for key in batch.keys():
            if key != 'z':
                batch[key] = batch[key][:, :max_episode_len]
        self.policy.learn(batch, max_episode_len, train_step, epsilon)

        if train_step > 0 and train_step % self.args.save_cycle == 0:
            print("\n开始保存模型", 'train_step:', train_step, 'save_cycle:', self.args.save_cycle)
            self.policy.save_model(train_step)
