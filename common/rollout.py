import copy
import numpy as np
import task_generator_fixed
from SD_rules import sd_rules_agent_wrapper


class RolloutWorker:
    def __init__(self, env, agents, args):
        self.env = env
        self.agents = agents
        self.args = args
        self.n_actions = args.n_actions
        self.n_agents = args.n_agents
        self.state_shape = args.state_shape
        self.obs_shape = args.obs_shape
        self.episode_limit = args.episode_limit
        self.epsilon = args.epsilon
        self.anneal_epsilon = args.anneal_epsilon
        self.min_epsilon = args.min_epsilon
        print('Init RolloutWorker')

    def generate_episode(self, evaluate=False):
        # 初始化
        step = 0
        episode_reward = 0
        terminated = False
        epsilon = 0 if evaluate else self.epsilon
        o, u, r, s, avail_u, u_onehot, terminate, padded = [], [], [], [], [], [], [], []
        last_action = np.zeros((self.args.n_agents, self.args.n_actions))
        self.agents.policy.init_hidden(1)
        self.env.reset()
        self.env.tasks_array = task_generator_fixed.generate_tasks()
        sd_episode_reward, sd_episode_time_on_road, sd_episode_time_wait, _ = sd_rules_agent_wrapper(self.env.tasks_array)
        self.env.set_sd_episode_time_wait(sd_episode_time_wait)

        while not terminated and step < self.episode_limit:
            actions = np.zeros(self.args.n_agents)
            avail_actions = [[] for _ in range(self.args.n_agents)]
            actions_onehot = np.zeros((self.args.n_agents, self.args.n_actions))
            self.env.renew_wait_time()
            state = self.env.get_state()

            for agent_id in range(self.n_agents):
                # action = 7  7 - 机器人正忙或者当前无任务可以执行，不进行新的分配
                self.env.renew_task_window(agent_id)
                avail_action = self.env.get_avail_actions(agent_id)
                self.env.obs4marl[agent_id] = self.env.get_agent_obs(agent_id, avail_action)
                action = self.agents.choose_action(self.env.obs4marl[agent_id], last_action[agent_id], agent_id,
                                                   avail_action, epsilon)
                action = int(action)
                if action != 7:
                    self.env.has_chosen_action(action, agent_id)

                # 生成关于动作的one-hot向量
                action_onehot = np.zeros(self.args.n_actions)
                action_onehot[action] = 1
                actions[agent_id] = action
                actions_onehot[agent_id] = action_onehot
                avail_actions[agent_id] = avail_action
                last_action[agent_id] = action_onehot

            # 保存观测、状态和动作信息
            o.append(copy.deepcopy(self.env.obs4marl))
            s.append(copy.deepcopy(state))
            u.append(np.reshape(actions, [self.n_agents, 1]))
            u_onehot.append(actions_onehot)
            avail_u.append(avail_actions)
            # 进入env，执行一个step
            # print(self.env.obs4marl)
            # print("avail_actions", avail_actions)
            # print("qmix_choose_action: ", actions)
            # print("SD_choose_action: ", actions_sd)
            reward, terminated = self.env.step(actions)
            # 保存奖励，终止信息
            r.append([reward])
            terminate.append([terminated])
            padded.append([0.])
            episode_reward += reward
            step += 1
        if self.args.epsilon_anneal_scale == 'episode':
            # print("epsilon:", epsilon)
            epsilon = epsilon - self.anneal_epsilon if epsilon > self.min_epsilon else self.min_epsilon

        episode_time_on_road, episode_time_wait = self.env.get_time()
        # last obs
        o.append(copy.deepcopy(self.env.obs4marl))
        s.append(copy.deepcopy(self.env.state4marl))
        o_next = o[1:]
        s_next = s[1:]
        o = o[:-1]
        s = s[:-1]
        # get avail_action for last obs，because target_q needs avail_action in training
        avail_actions = []

        for agent_id in range(self.n_agents):
            avail_action = self.env.get_avail_actions(agent_id)
            avail_actions.append(avail_action)
        avail_u.append(avail_actions)
        avail_u_next = avail_u[1:]
        avail_u = avail_u[:-1]
        # if step < self.episode_limit，padding
        for i in range(step, self.episode_limit):
            o.append(np.zeros((self.n_agents, self.obs_shape)))
            u.append(np.zeros([self.n_agents, 1]))
            s.append(np.zeros(self.state_shape))
            r.append([0.])
            o_next.append(np.zeros((self.n_agents, self.obs_shape)))
            s_next.append(np.zeros(self.state_shape))
            u_onehot.append(np.zeros((self.n_agents, self.n_actions)))
            avail_u.append(np.zeros((self.n_agents, self.n_actions)))
            avail_u_next.append(np.zeros((self.n_agents, self.n_actions)))
            padded.append([1.])
            terminate.append([1.])  # 如果没有padding的情况下是没有terminal的

        episode = dict(o=o.copy(),
                       s=s.copy(),
                       u=u.copy(),
                       r=r.copy(),
                       avail_u=avail_u.copy(),
                       o_next=o_next.copy(),
                       s_next=s_next.copy(),
                       avail_u_next=avail_u_next.copy(),
                       u_onehot=u_onehot.copy(),
                       padded=padded.copy(),
                       terminated=terminate.copy()
                       )
        for key in episode.keys():
            episode[key] = np.array([episode[key]], dtype=object)
        if not evaluate:
            self.epsilon = epsilon
        # print('episode_reward:', episode_reward)
        return episode, episode_reward, episode_time_on_road, episode_time_wait, terminated

    def evaluate_episode(self, tasks):
        self.env.reset()
        self.env.tasks_array = tasks
        sd_episode_reward, sd_episode_time_on_road, sd_episode_time_wait, _ = sd_rules_agent_wrapper(self.env.tasks_array)
        self.env.set_sd_episode_time_wait(sd_episode_time_wait)
        # print(self.env.tasks_array)
        terminated = False
        step = 0
        epsilon = 0
        episode_reward = 0  # 累积奖励
        last_action = np.zeros((self.args.n_agents, self.args.n_actions))
        self.agents.policy.init_hidden(1)
        while not terminated and step < self.episode_limit:
            actions = np.zeros(self.args.n_agents)
            avail_actions = [[] for _ in range(self.args.n_agents)]
            actions_onehot = np.zeros((self.args.n_agents, self.args.n_actions))
            self.env.renew_wait_time()
            state = self.env.get_state()
            for agent_id in range(self.n_agents):

                self.env.renew_task_window(agent_id)
                avail_action = self.env.get_avail_actions(agent_id)
                self.env.obs4marl[agent_id] = self.env.get_agent_obs(agent_id, avail_action)
                action = self.agents.choose_action(self.env.obs4marl[agent_id], last_action[agent_id], agent_id,
                                                   avail_action, epsilon)
                action = int(action)
                # print("robot: ", agent_id, "avail_action: ", avail_action, "qmix_choose_action: ", action)

                if action != 7:
                    self.env.has_chosen_action(action, agent_id)

                # 生成关于动作的one-hot向量
                action_onehot = np.zeros(self.args.n_actions)
                action_onehot[action] = 1
                actions[agent_id] = action
                actions_onehot[agent_id] = action_onehot
                avail_actions[agent_id] = avail_action
                last_action[agent_id] = action_onehot

            # 进入env，执行一个step
            reward, terminated = self.env.step(actions)
            episode_reward += reward
            step += 1
        episode_time_on_road, episode_time_wait = self.env.get_time()

        return episode_reward, episode_time_on_road, episode_time_wait, terminated
