"""
训练时需要的参数
"""
import argparse


def get_common_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--replay_dir', type=str, default=r'', help='absolute path to save the replay')
    parser.add_argument('--model_dir', type=str, default='./MARL/model', help='model directory of the policy')
    parser.add_argument('--result_dir', type=str, default='./result', help='result directory of the policy')
    parser.add_argument('--load_model', type=bool, default=False, help='whether to load the pretrained model')
    parser.add_argument('--alg', type=str, default='qmix', help='the algorithm to train the agent')
    parser.add_argument('--last_action', type=bool, default=False, help='whether to use the last action to choose action')
    parser.add_argument('--reuse_network', type=bool, default=False, help='whether to use one network for all agents')
    parser.add_argument('--gamma', type=float, default=0.995, help='discount factor')
    parser.add_argument('--optimizer', type=str, default="RMS", help='optimizer')
    parser.add_argument('--evaluate_epoch', type=int, default=20, help='number of the epoch to evaluate the agent')
    parser.add_argument('--learn', type=bool, default=True, help='whether to train the model')
    parser.add_argument('--cuda', type=bool, default=True, help='whether to use the GPU')
    parser.add_argument('--map', type=str, default="Schedule", help='map name')
    parser.add_argument('--havelook', type=bool, default=True, help='whether to have a look of the saruo')
    args = parser.parse_args()
    return args


# arguments of q-mix
def get_mixer_args(args):
    # network
    args.rnn_hidden_dim = 64
    args.qmix_hidden_dim = 32
    args.two_hyper_layers = False
    args.hyper_hidden_dim = 64
    args.lr = 1.5e-4
    args.epsilon = 0.2  #
    args.min_epsilon = 0.01  #
    anneal_steps = 1500   #
    args.anneal_epsilon = (args.epsilon - args.min_epsilon) / anneal_steps
    args.epsilon_anneal_scale = 'episode'
    # the number of the epoch to train the agent
    args.n_epoch = 15000
    # the number of the episodes in one epoch
    args.n_episodes = 5
    # the number of the train steps in one epoch
    args.train_steps = 2
    # # how often to evaluate
    args.evaluate_cycle = 20
    # experience replay
    args.batch_size = 128
    args.buffer_size = int(1e4)
    # how often to save the model
    args.save_cycle = 50
    # how often to update the target_net
    args.target_update_cycle = 200
    # prevent gradient explosion
    args.grad_norm_clip = 10
    return args


