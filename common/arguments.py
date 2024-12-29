"""
训练时需要的参数
"""
import argparse


def get_common_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--replay_dir', type=str, default=r'', help='absolute path to save the replay')
    parser.add_argument('--model_dir', type=str, default='./model', help='model directory of the policy')
    parser.add_argument('--result_dir', type=str, default='./result', help='result directory of the policy')
    parser.add_argument('--load_model', type=bool, default=False, help='whether to load the pretrained model')
    parser.add_argument('--alg', type=str, default='qmix', help='the algorithm to train the agent')
    parser.add_argument('--last_action', type=bool, default=False, help='whether to use the last action to choose action')
    parser.add_argument('--reuse_network', type=bool, default=False, help='whether to use one network for all agents')
    parser.add_argument('--n_steps', type=int, default=2000000, help='total time steps')
    parser.add_argument('--gamma', type=float, default=0.995, help='discount factor')
    parser.add_argument('--optimizer', type=str, default="RMS", help='optimizer')
    parser.add_argument('--evaluate_epoch', type=int, default=20, help='number of the epoch to evaluate the agent')
    parser.add_argument('--learn', type=bool, default=True, help='whether to train the model')
    parser.add_argument('--cuda', type=bool, default=True, help='whether to use the GPU')
    parser.add_argument('--map', type=str, default="Schedule", help='map name')
    parser.add_argument('--log_step_data', type=bool, default=False, help='Log step data for debugging')
    parser.add_argument("--use_tensorboard", type=bool, default=True, help="Enable TensorBoard logging")
    parser.add_argument("--run_name", type=str, default="default_run", help="Name of the current run")
    args = parser.parse_args()
    return args


# arguments of q-mix
def get_mixer_args(args):
    # network
    args.rnn_hidden_dim = 64
    args.qmix_hidden_dim = 32
    args.two_hyper_layers = False
    args.hyper_hidden_dim = 64
    args.lr = 5e-4
    args.epsilon = 1  #
    args.min_epsilon = 0.05  #
    anneal_steps = 350000   #
    args.anneal_epsilon = (args.epsilon - args.min_epsilon) / anneal_steps
    args.epsilon_anneal_scale = 'step'
    # the number of the epoch to train the agent
    args.n_epoch = 10000
    # the number of the episodes in one epoch
    args.n_episodes = 5
    # the number of the train steps in one epoch
    args.train_steps = 1
    # # how often to evaluate
    args.evaluate_cycle = 50
    # experience replay
    args.batch_size = 32
    args.buffer_size = int(5e3)
    # how often to save the model
    args.save_cycle = 50
    # how often to update the target_net
    args.target_update_cycle = 200
    # prevent gradient explosion
    args.grad_norm_clip = 10
    return args


