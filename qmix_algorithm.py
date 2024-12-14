from test_task import load_tasks_from_file
from common.rollout import RolloutWorker
from agent.agent import Agents
from common.arguments import get_common_args, get_mixer_args
from environment import ScheduleEnv
import sys
from os.path import dirname, abspath
sys.path.append(dirname(dirname(abspath(__file__))))


def qmix_rules_agent_wrapper(tasks, rollout):
    episode_reward, episode_time_on_road, episode_time_wait, terminated = rollout.evaluate_episode(tasks)
    return episode_reward, episode_time_on_road, episode_time_wait, terminated


if __name__ == "__main__":
    args = get_common_args()
    args = get_mixer_args(args)
    env = ScheduleEnv()
    env_info = env.get_env_info()
    args.n_actions = env_info["n_actions"]
    args.n_agents = env_info["n_agents"]
    args.state_shape = env_info["state_shape"]
    args.obs_shape = env_info["obs_shape"]
    args.episode_limit = env_info["episode_limit"]
    agents = Agents(args)
    rolloutWorker = RolloutWorker(env, agents, args)
    test_tasks = load_tasks_from_file('task/test_tasks.pkl')
    episode_time_wait_list = []
    # for i in range(len(test_tasks)):
    for i in range(1):
        reward, time_on_road,  time_wait, done = qmix_rules_agent_wrapper(test_tasks[i], rolloutWorker)
        print('task_id:', i, 'episode_time_on_road: ', time_on_road, 'episode_time_wait:', time_wait, "done:", done)
        episode_time_wait_list.append(time_wait)
    average_wait = sum(episode_time_wait_list)/len(episode_time_wait_list)
    print('qmix_algorithm average time_wait:', average_wait)



