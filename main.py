from environment import ScheduleEnv
from runner import Runner
from common.arguments import get_common_args, get_mixer_args
import sys
from os.path import dirname, abspath
sys.path.append(dirname(dirname(abspath(__file__))))


def marl_agent_wrapper():

    args = get_common_args()
    args = get_mixer_args(args)
    env = ScheduleEnv()
    env_info = env.get_env_info()
    args.n_actions = env_info["n_actions"]
    args.n_agents = env_info["n_agents"]
    args.state_shape = env_info["state_shape"]
    args.obs_shape = env_info["obs_shape"]
    args.episode_limit = env_info["episode_limit"]
    print("是否加载模型（测试必须）：", args.load_model, "是否训练：", args.learn)
    runner = Runner(env, args)

    if args.learn:
        runner.run()


if __name__ == "__main__":
    marl_agent_wrapper()
