import argparse

"""
Here are the param for the training

"""

def get_args():
    parser = argparse.ArgumentParser()
    
    # Original DDPG args
    parser.add_argument('--algo', type=str, default='ddpg', help='the reinforcement learning algorithm to use')
    parser.add_argument('--env-name', type=str, default='FetchReach-v1', help='the environment name')
    parser.add_argument('--n-epochs', type=int, default=50, help='the number of epochs to train the agent')
    parser.add_argument('--n-cycles', type=int, default=50, help='the times to collect samples per epoch')
    parser.add_argument('--n-batches', type=int, default=40, help='the times to update the network')
    parser.add_argument('--save-interval', type=int, default=5, help='the interval that save the trajectory')
    parser.add_argument('--seed', type=int, default=123, help='random seed')
    parser.add_argument('--num-workers', type=int, default=1, help='the number of cpus to collect samples')
    parser.add_argument('--replay-strategy', type=str, default='future', help='the HER strategy')
    parser.add_argument('--clip-return', type=float, default=50, help='if clip the returns')
    parser.add_argument('--save-dir', type=str, default='rrc_example_package/her/saved_models/', help='the path to save the models')
    parser.add_argument('--exp-dir', type=str, default='test', help='the path to save the models')
    parser.add_argument('--noise-eps', type=float, default=0.2, help='noise eps')
    parser.add_argument('--random-eps', type=float, default=0.3, help='random eps')
    parser.add_argument('--buffer-size', type=int, default=int(1e6), help='the size of the buffer')
    parser.add_argument('--replay-k', type=int, default=4, help='ratio to be replace')
    parser.add_argument('--clip-obs', type=float, default=200, help='the clip ratio')
    parser.add_argument('--batch-size', type=int, default=256, help='the sample batch size')
    parser.add_argument('--gamma', type=float, default=0.99, help='the discount factor')
    parser.add_argument('--action-l2', type=float, default=1, help='l2 reg')
    parser.add_argument('--lr-actor', type=float, default=0.001, help='the learning rate of the actor')
    parser.add_argument('--lr-critic', type=float, default=0.001, help='the learning rate of the critic')
    parser.add_argument('--polyak-ddpg', type=float, default=0.95, help='the average coefficient')
    parser.add_argument('--n-test-rollouts', type=int, default=10, help='the number of tests')
    parser.add_argument('--clip-range', type=float, default=5, help='the clip range')
    parser.add_argument('--demo-length', type=int, default=20, help='the demo length')
    parser.add_argument('--cuda', action='store_true', help='if use gpu do the acceleration')
    parser.add_argument('--num-rollouts-per-mpi', type=int, default=2, help='the rollouts per mpi')
    parser.add_argument('--load-pretrained', type=int, default=0, help='whether to load in a pretrained model')
    parser.add_argument('--pretrained-dir', type=str, default='test/test.pt', help='pretrained model path')
    
    # Additional SAC args
    parser.add_argument('--hidden-size', type=int, default=256, help='hidden size of networks')
    parser.add_argument('--n-hiddens', type=int, default=2, help='number of hidden sizes')
    parser.add_argument('--alpha', type=float, default=0.2, help='Entropy regularization coefficient. (Equivalent to inverse of reward scale in the original SAC paper.)')
    parser.add_argument('--polyak-sac', type=float, default=0.995, help='Interpolation factor in polyak averaging for target networks.')
    parser.add_argument('--start-epochs', type=int, default=2, help='number of epochs to take random actions at start of training')
    parser.add_argument('--save-freq', type=int, default=10, help='per epoch save rate')
    parser.add_argument('--save-models', type=int, default=1, help='whether to save models') # 1=True, 0=False
    
    # Dice env related args
    parser.add_argument('--step-size', type=int, default=50, help='frame skip')
    parser.add_argument('--enable-cameras', type=int, default=0, help='whether to render and use camera images') # 1=True, 0=False
    parser.add_argument('--num-dice', type=int, default=5, help='number of dice to train with')
    parser.add_argument('--include-dice-velocity', type=int, default=1, help='whether to include dice velocities in observations') # 1=True, 0=False
    parser.add_argument('--include-dice-orient', type=int, default=1, help='whether to include dice orientation in observations') # 1=True, 0=False
    parser.add_argument('--distance-threshold', type=float, default=0.01, help='threshold for sparse distance based reward')
    parser.add_argument('--max-steps', type=int, default=50, help='max steps per episode')
    parser.add_argument('--seperate-her', type=int, default=0, help='whether to apply HER individually to each dice goal')
    parser.add_argument('--scale-dice', type=float, default=1, help='scale size of the dice')
    parser.add_argument('--single-focus', type=int, default=0, help='whether to focus on a single dice and goal only')
    parser.add_argument('--increment-dice', type=int, default=0, help='periodically increase the number of dice by this amount')
    parser.add_argument('--increment-freq', type=int, default=10, help='epoch frequency to increment number of dice')
    parser.add_argument('--update-wait', type=int, default=2, help='epochs to collect data without update after incrementing num dice')
    parser.add_argument('--max-dice', type=int, default=10, help='max num dice to increment to')
    
    # Intrinsic reward/dynamics args
    parser.add_argument('--include-ri', action='store_true', help='whether to use intrinsic rewards')
    parser.add_argument('--epochs-per-dy-update', type=int, default=5, help='frequency to update dynamics')
    parser.add_argument('--ensemble-size', type=int, default=5, help='size of dynamics model ensemble')
    parser.add_argument('--dynamics-hiddens', type=int, default=2, help='no. hidden layers in dynamics model')
    parser.add_argument('--dynamics-hsize', type=int, default=512, help='size of dynamics hidden layers')
    parser.add_argument('--dynamics-steps', type=int, default=1000, help='update steps per update for dynamics model')
    parser.add_argument('--dynamics-batch-size', type=int, default=512, help='size of batches for dynamics updates')
    parser.add_argument('--scale-ri', type=float, default=10, help='scale intrinsic rewards')
    parser.add_argument('--clip-ri', type=float, default=0.8, help='upper clip intrinsic rewards')

    args = parser.parse_args()

    return args
