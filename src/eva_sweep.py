import argparse
import numpy as np
import torch
from network_env import RoadWorld
from utils.load_data import load_test_traj, ini_od_dist, load_path_feature, load_link_feature, minmax_normalization
from model.policy import PolicyCNN
from model.value import ValueCNN
from model.discriminator import DiscriminatorAIRLCNN
from utils.evaluation import evaluate_model
import pandas as pd

def load_model(model_path, device, env, path_feature_pad, edge_feature_pad):
    # Load the model dictionary
    model_dict = torch.load(model_path, map_location=device)
    hyperparameters = model_dict.get('Hyperparameters', {})
    gamma = hyperparameters.get('gamma', 0.99)  # Use the saved gamma if available

    # Initialize models with the correct dimensions and hyperparameters
    policy_net = PolicyCNN(env.n_actions, env.policy_mask, env.state_action,
                           path_feature_pad, edge_feature_pad,
                           path_feature_pad.shape[-1] + edge_feature_pad.shape[-1] + 1,
                           env.pad_idx).to(device)
    value_net = ValueCNN(path_feature_pad, edge_feature_pad,
                         path_feature_pad.shape[-1] + edge_feature_pad.shape[-1]).to(device)
    discrim_net = DiscriminatorAIRLCNN(env.n_actions, gamma, env.policy_mask,
                                       env.state_action, path_feature_pad, edge_feature_pad,
                                       path_feature_pad.shape[-1] + edge_feature_pad.shape[-1] + 1,
                                       path_feature_pad.shape[-1] + edge_feature_pad.shape[-1],
                                       env.pad_idx).to(device)

    policy_net.load_state_dict(model_dict['Policy'])
    value_net.load_state_dict(model_dict['Value'])
    discrim_net.load_state_dict(model_dict['Discrim'])

    print("Loaded model with hyperparameters:")
    for key, value in hyperparameters.items():
        print(f"{key}: {value}")

    return policy_net, value_net, discrim_net

def evaluate_only(model_path, cv_index=0):
    device = torch.device('cpu')
    
    # Path settings
    edge_p = "../data/base/edge.txt"
    network_p = "../data/base/transit.npy"
    path_feature_p = "../data/base/feature_od.npy"
    test_p = "../data/base/cross_validation/test_CV%d.csv" % cv_index

    # Initialize environment
    od_list, od_dist = ini_od_dist(test_p)
    env = RoadWorld(network_p, edge_p, pre_reset=(od_list, od_dist))

    # Load features and normalize
    path_feature, path_max, path_min = load_path_feature(path_feature_p)
    edge_feature, link_max, link_min = load_link_feature(edge_p)
    path_feature = minmax_normalization(path_feature, path_max, path_min)
    path_feature_pad = np.zeros((env.n_states, env.n_states, path_feature.shape[2]))
    path_feature_pad[:path_feature.shape[0], :path_feature.shape[1], :] = path_feature
    edge_feature = minmax_normalization(edge_feature, link_max, link_min)
    edge_feature_pad = np.zeros((env.n_states, edge_feature.shape[1]))
    edge_feature_pad[:edge_feature.shape[0], :] = edge_feature

    # Load the model
    policy_net, value_net, discrim_net = load_model(model_path, device, env, path_feature_pad, edge_feature_pad)

    # Load test trajectories
    test_trajs, test_od = load_test_traj(test_p)

    # Evaluate the model
    evaluate_model(test_od, test_trajs, policy_net, env)

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Evaluate AIRL Model')
    parser.add_argument('--model_path', type=str, required=True, help='Path to the model file')
    parser.add_argument('--cv_index', type=int, default=0, help='Cross-validation index (default: 0)')
    args = parser.parse_args()

    evaluate_only(args.model_path, args.cv_index)

# python eva.py --model_path PATH_TO_YOUR_MODEL_FILE --cv_index CV_INDEX
# python eva_sweep.py --model_path /Users/mochi/Documents/Study/rcm/trained_models/base/airl_lr0_000240151566904156_bs128_gamma0_95_tau0_95_clip0_2_epoch20_runz6irxkua.pt --cv_index 0

#  /Users/mochi/Documents/Study/rcm/trained_models/base 
