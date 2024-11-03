# rl_training.py

import numpy as np
from model.policy import PolicyCNN
from model.value import ValueCNN
import torch.nn.functional as F
import torch
from torch import nn
import math
import time
from network_env import RoadWorld
from core.ppo import ppo_step
from core.common import estimate_advantages
from core.agent import Agent
from utils.torch import to_device
from utils.evaluation import evaluate_model, evaluate_log_prob, evaluate_train_edit_dist
from utils.load_data import ini_od_dist, load_path_feature, load_link_feature, \
    minmax_normalization, load_train_sample, load_test_traj

import csv
import pandas as pd

torch.backends.cudnn.enabled = False

def update_params_rl(batch, i_iter):
    states = torch.from_numpy(np.stack(batch.state)).long().to(device)
    masks = torch.from_numpy(np.stack(batch.mask)).long().to(device)
    bad_masks = torch.from_numpy(np.stack(batch.bad_mask)).long().to(device)
    actions = torch.from_numpy(np.stack(batch.action)).long().to(device)
    destinations = torch.from_numpy(np.stack(batch.destination)).long().to(device)
    next_states = torch.from_numpy(np.stack(batch.next_state)).long().to(device)
    rewards = torch.from_numpy(np.stack(batch.reward)).float().to(device)

    with torch.no_grad():
        values = value_net(states, destinations)
        next_values = value_net(next_states, destinations)
        fixed_log_probs = policy_net.get_log_prob(states, destinations, actions)

    advantages, returns = estimate_advantages(rewards, masks, bad_masks, values, next_values, gamma, tau, device)

    """perform mini-batch PPO update"""
    value_loss, policy_loss = 0, 0
    optim_iter_num = int(math.ceil(states.shape[0] / optim_batch_size))
    for _ in range(optim_epochs):
        value_loss, policy_loss = 0, 0
        perm = np.arange(states.shape[0])
        np.random.shuffle(perm)
        perm = torch.LongTensor(perm).to(device)
        states, destinations, actions, returns, advantages, fixed_log_probs = \
            states[perm].clone(), destinations[perm].clone(), actions[perm].clone(), returns[perm].clone(), advantages[
                perm].clone(), fixed_log_probs[perm].clone()
        for i in range(optim_iter_num):
            ind = slice(i * optim_batch_size, min((i + 1) * optim_batch_size, states.shape[0]))
            states_b, destinations_b, actions_b, advantages_b, returns_b, fixed_log_probs_b = \
                states[ind], destinations[ind], actions[ind], advantages[ind], returns[ind], fixed_log_probs[ind],
            batch_value_loss, batch_policy_loss = ppo_step(policy_net, value_net, optimizer_policy, optimizer_value, 1,
                                                           states_b, destinations_b, actions_b, returns_b,
                                                           advantages_b, fixed_log_probs_b, clip_epsilon, l2_reg,
                                                           max_grad_norm)
            value_loss += batch_value_loss.item()
            policy_loss += batch_policy_loss.item()
    return value_loss, policy_loss

def save_model(model_path):
    policy_statedict = policy_net.state_dict()
    value_statedict = value_net.state_dict()
    outdict = {"Policy": policy_statedict,
               "Value": value_statedict}
    torch.save(outdict, model_path)

def load_model(model_path):
    model_dict = torch.load(model_path)
    policy_net.load_state_dict(model_dict['Policy'])
    print("Policy Model loaded Successfully")
    value_net.load_state_dict(model_dict['Value'])
    print("Value Model loaded Successfully")

def main_loop():
    best_edit = 1.0
    # Open a CSV file for logging
    with open('training_log.csv', 'w', newline='') as csvfile:
        fieldnames = ['Iteration', 'Elapsed Time', 'Value Loss', 'Policy Loss', 'Edit Distance', 'Best Edit Distance']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()

        for i_iter in range(1, max_iter_num + 1):
            batch, log = agent.collect_samples(min_batch_size, mean_action=False)
            value_loss, policy_loss = update_params_rl(batch, i_iter)
            if i_iter % log_interval == 0:
                elapsed_time = time.time() - start_time
                print(f"Iteration {i_iter}/{max_iter_num} | Elapsed Time: {elapsed_time:.2f}s")
                print(f"Value Loss: {value_loss:.4f} | Policy Loss: {policy_loss:.4f}")
                
                learner_trajs = agent.collect_routes_with_OD(test_od, mean_action=True)
                edit_dist = evaluate_train_edit_dist(test_trajs, learner_trajs)
                print(f"Edit Distance: {edit_dist:.4f} | Best Edit Distance: {best_edit:.4f}")
                
                if edit_dist < best_edit:
                    best_edit = edit_dist
                    save_model(model_p)
                    print("Model saved.")
                
                print("---")

                # Write the iteration details to the CSV file
                writer.writerow({
                    'Iteration': i_iter,
                    'Elapsed Time': elapsed_time,
                    'Value Loss': value_loss,
                    'Policy Loss': policy_loss,
                    'Edit Distance': edit_dist,
                    'Best Edit Distance': best_edit
                })

if __name__ == '__main__':
    log_std = -0.0  # Log std for the policy
    gamma = 0.99  # Discount factor
    tau = 0.95  # GAE parameter
    l2_reg = 1e-3  # L2 regularization (not used in the model)
    learning_rate = 3e-4  # Learning rate
    clip_epsilon = 0.2  # Clipping epsilon for PPO
    num_threads = 4  # Number of threads for agent
    min_batch_size = 8192  # Minimal batch size per PPO update
    eval_batch_size = 8192  # Minimal batch size for evaluation
    log_interval = 10  # Interval between training status logs
    save_mode_interval = 50  # Interval between saving model
    max_grad_norm = 10  # Max grad norm for PPO updates
    seed = 1  # Random seed for parameter initialization
    optim_epochs = 10  # Optimization epoch number for PPO
    optim_batch_size = 64  # Optimization batch size for PPO
    cv = 0  # Cross-validation process [0, 1, 2, 3, 4]
    size = 10000  # Size of training data [100, 1000, 10000]
    max_iter_num = 2000  # Maximal number of main iterations
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    """Environment"""
    edge_p = "../data/base/edge.txt"
    network_p = "../data/base/transit.npy"
    path_feature_p = "../data/base/feature_od.npy"
    train_p = "../data/base/cross_validation/train_CV%d_size%d.csv" % (cv, size)
    test_p = "../data/base/cross_validation/test_CV%d.csv" % cv
    model_p = "../trained_models/base/rl_CV%d_size%d.pt" % (cv, size)

    """Initialize road environment"""
    od_list, od_dist = ini_od_dist(train_p)
    env = RoadWorld(network_p, edge_p, pre_reset=(od_list, od_dist))
    """Load path-level and link-level features"""
    path_feature, path_max, path_min = load_path_feature(path_feature_p)
    edge_feature, link_max, link_min = load_link_feature(edge_p)
    path_feature = minmax_normalization(path_feature, path_max, path_min)
    path_feature_pad = np.zeros((env.n_states, env.n_states, path_feature.shape[2]))
    path_feature_pad[:path_feature.shape[0], :path_feature.shape[1], :] = path_feature
    edge_feature = minmax_normalization(edge_feature, link_max, link_min)
    edge_feature_pad = np.zeros((env.n_states, edge_feature.shape[1]))
    edge_feature_pad[:edge_feature.shape[0], :] = edge_feature
    """Seeding"""
    np.random.seed(seed)
    torch.manual_seed(seed)
    """Define actor and critic"""
    policy_net = PolicyCNN(env.n_actions, env.policy_mask, env.state_action,
                           path_feature_pad, edge_feature_pad,
                           path_feature_pad.shape[-1] + edge_feature_pad.shape[-1] + 1,
                           env.pad_idx).to(device)
    value_net = ValueCNN(path_feature_pad, edge_feature_pad,
                         path_feature_pad.shape[-1] + edge_feature_pad.shape[-1]).to(device)
    policy_net.to_device(device)
    value_net.to_device(device)
    optimizer_policy = torch.optim.Adam(policy_net.parameters(), lr=learning_rate)
    optimizer_value = torch.optim.Adam(value_net.parameters(), lr=learning_rate)
    """Load expert trajectories for evaluation (optional)"""
    test_trajs, test_od = load_train_sample(train_p)
    """Create agent"""
    agent = Agent(env, policy_net, device, custom_reward=None, num_threads=num_threads)
    print('Agent constructed successfully...')
    """Train model"""
    start_time = time.time()
    main_loop()
    print('Training time:', time.time() - start_time)
    """Evaluate model"""
    load_model(model_p)
    test_trajs, test_od = load_test_traj(test_p)
    start_time = time.time()
    evaluate_model(test_od, test_trajs, policy_net, env)
    print('Evaluation time:', time.time() - start_time)
    """Evaluate log probability (optional)"""
    test_trajs = env.import_demonstrations_step(test_p)
    evaluate_log_prob(test_trajs, policy_net)
