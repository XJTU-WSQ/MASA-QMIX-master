import copy
import json
import numpy as np
from task import task_generator


class RolloutWorker:
    def __init__(self, env, agents, args):
        self.env = env
        self.agents = agents
        self.episode_limit = args.episode_limit
        self.n_actions = args.n_actions
        self.n_agents = args.n_agents
        self.state_shape = args.state_shape
        self.obs_shape = args.obs_shape
        self.args = args

        self.epsilon = args.epsilon
        self.anneal_epsilon = args.anneal_epsilon
        self.min_epsilon = args.min_epsilon
        print('Init RolloutWorker')

    def generate_episode(self, episode_num=None, evaluate=False):
        episode_data = []
        if self.args.replay_dir != '' and evaluate and episode_num == 0:  # prepare for save replay of evaluation
            self.env.close()
        o, u, r, s, avail_u, u_onehot, terminate, padded = [], [], [], [], [], [], [], []
        self.env.reset()
        terminated = False
        step = 0
        episode_reward = 0
        self.agents.policy.init_hidden(1)
        last_action = np.zeros((self.args.n_agents, self.args.n_actions))
        self.env.tasks_array = task_generator.generate_tasks()

        # epsilon
        epsilon = 0 if evaluate else self.epsilon
        if self.args.epsilon_anneal_scale == 'episode':
            epsilon = epsilon - self.anneal_epsilon if epsilon > self.min_epsilon else epsilon

        # 初始化统计量
        total_conflicts = 0
        total_task_rewards = 0
        total_wait_penalty = 0
        total_service_cost_penalty = 0

        while not terminated and step < self.episode_limit:

            self.env.update_task_window()
            self.env.renew_wait_time()
            obs = self.env.get_obs()
            state = self.env.get_state()
            actions, avail_actions, actions_onehot = [], [], []
            for agent_id in range(self.n_agents):

                avail_action = self.env.get_avail_agent_actions(agent_id)
                action = self.agents.choose_action(
                    obs[agent_id], agent_id, avail_action, epsilon
                )

                # generate onehot vector of th action
                action_onehot = np.zeros(self.args.n_actions)
                action_onehot[action] = 1
                actions.append(np.int(action))
                actions_onehot.append(action_onehot)
                avail_actions.append(avail_action)
                last_action[agent_id] = action_onehot

            reward, terminated, info = self.env.step(actions)

            # 累积统计量
            total_conflicts += info["conflict_count"]
            total_task_rewards += info["task_rewards"]
            total_wait_penalty += info["total_wait_penalty"]
            total_service_cost_penalty += info["total_service_cost_penalty"]
            # 如果开启数据记录，将数据存入缓冲区
            if self.args.log_step_data:
                episode_data.append({
                    "step": step,
                    "state": convert_to_native(state),
                    "obs": convert_to_native(obs),
                    "actions": convert_to_native(actions),
                    "reward": convert_to_native(reward),
                    "avail_actions": convert_to_native(avail_actions),
                    "robots_state": convert_to_native(info["robots_state"]),
                    "task_window": convert_to_native(info["task_window"]),
                    "conflict_count": convert_to_native(info["conflict_count"]),
                    "task_rewards": convert_to_native(info["task_rewards"]),
                    "total_service_cost_penalty": convert_to_native(info["total_service_cost_penalty"]),
                    "done": convert_to_native(info["done"])
                })
            # 保存观测、状态和动作信息
            o.append(obs)
            s.append(state)
            u.append(np.reshape(actions, [self.n_agents, 1]))
            u_onehot.append(actions_onehot)
            avail_u.append(avail_actions)
            r.append([reward])
            terminate.append([terminated])
            padded.append([0.])
            episode_reward += reward
            step += 1

            if self.args.epsilon_anneal_scale == 'step':
                epsilon = epsilon - self.anneal_epsilon if epsilon > self.min_epsilon else epsilon

        # last obs
        obs = self.env.get_obs()
        state = self.env.get_state()
        o.append(obs)
        s.append(state)
        o_next = o[1:]
        s_next = s[1:]
        o = o[:-1]
        s = s[:-1]
        # get avail_action for last obs，because target_q needs avail_action in training
        avail_actions = []
        for agent_id in range(self.n_agents):
            avail_action = self.env.get_avail_agent_actions(agent_id)
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
        if evaluate and episode_num == self.args.evaluate_epoch - 1 and self.args.replay_dir != '':
            self.env.save_replay()
            self.env.close()
        if terminated or step == self.episode_limit:
            if self.args.log_step_data:
                with open(f"./episode_logs/episode_{episode_num}.json", "w", encoding="utf-8") as f:
                    json.dump(episode_data, f, indent=4)
                print(f"Episode {episode_num} data saved to episode_{episode_num}.json")

        total_wait_time = self.env.total_time_wait
        total_completed_tasks = sum(self.env.tasks_completed)
        total_tasks = len(self.env.tasks_array)
        completion_rate = total_completed_tasks / total_tasks
        # 构建统计量
        stats = {
            "conflicts": total_conflicts,
            "wait_time": total_wait_time,
            "completed_tasks": total_completed_tasks,
            "completion_rate": completion_rate,
            "episode_reward": episode_reward,
            "task_rewards": total_task_rewards,
            "wait_penalty": total_wait_penalty,
            "service_cost_penalty": total_service_cost_penalty,
        }

        return episode, episode_reward, terminated, stats


def convert_to_native(obj):
    """
    将 numpy 类型转换为 Python 原生类型，确保 JSON 序列化兼容。
    """
    if isinstance(obj, np.ndarray):
        return obj.tolist()  # 转为列表
    elif isinstance(obj, (np.bool_, bool)):
        return bool(obj)  # 转为 Python 布尔类型
    elif isinstance(obj, (np.integer, int)):
        return int(obj)  # 转为 Python 整数
    elif isinstance(obj, (np.floating, float)):
        return float(obj)  # 转为 Python 浮点数
    elif isinstance(obj, list):
        return [convert_to_native(i) for i in obj]  # 递归处理列表
    elif isinstance(obj, dict):
        return {k: convert_to_native(v) for k, v in obj.items()}  # 递归处理字典
    else:
        return obj  # 返回原始值
