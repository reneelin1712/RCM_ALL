import pandas as pd
import numpy as np
import torch
import torch.nn.functional as F
import os

df = pd.read_excel('same_OD_different_paths_processed.xlsx')

df['Highest_Reward_Index'] = None
df['Highest_Reward_Value'] = None
df['Lowest_Reward_Index'] = None
df['Lowest_Reward_Value'] = None
df['Diff_Index_Reward'] = None
df['Same_Again_Index_Reward'] = None

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Paths to your data and model (adjust as necessary)
model_path = "../trained_models/base/bleu90.pt"
edge_p = "../data/base/edge.txt"
network_p = "../data/base/transit.npy"
path_feature_p = "../data/base/feature_od.npy"
test_p = "../data/base/cross_validation/test_CV0.csv"

# Import your custom modules (ensure these are accessible)
from network_env import RoadWorld
from utils.load_data import (
    ini_od_dist,
    load_path_feature,
    load_link_feature,
    minmax_normalization,
)
from model.policy import PolicyCNN
from model.value import ValueCNN
from model.discriminator import DiscriminatorAIRLCNN

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
def load_model(model_path, device, env, path_feature_pad, edge_feature_pad):
    gamma = 0.95  # discount factor
    policy_net = PolicyCNN(
        env.n_actions,
        env.policy_mask,
        env.state_action,
        path_feature_pad,
        edge_feature_pad,
        path_feature_pad.shape[-1] + edge_feature_pad.shape[-1] + 1,
        env.pad_idx,
    ).to(device)
    value_net = ValueCNN(
        path_feature_pad,
        edge_feature_pad,
        path_feature_pad.shape[-1] + edge_feature_pad.shape[-1],
    ).to(device)
    discriminator_net = DiscriminatorAIRLCNN(
        env.n_actions,
        gamma,
        env.policy_mask,
        env.state_action,
        path_feature_pad,
        edge_feature_pad,
        path_feature_pad.shape[-1] + edge_feature_pad.shape[-1] + 1,
        path_feature_pad.shape[-1] + edge_feature_pad.shape[-1],
        env.pad_idx,
    ).to(device)

    model_dict = torch.load(model_path, map_location=device)
    policy_net.load_state_dict(model_dict['Policy'])
    value_net.load_state_dict(model_dict['Value'])
    discriminator_net.load_state_dict(model_dict['Discrim'])

    policy_net.eval()
    value_net.eval()
    discriminator_net.eval()

    return policy_net, value_net, discriminator_net

# Load the model
policy_net, value_net, discriminator_net = load_model(
    model_path, device, env, path_feature_pad, edge_feature_pad
)

def get_cnn_input(policy_net, state, des, device):
    state = torch.tensor([state], dtype=torch.long).to(device)
    des = torch.tensor([des], dtype=torch.long).to(device)
    # Process features to get the CNN input
    input_data = policy_net.process_features(state, des)
    return input_data

def compute_reward(discriminator_net, state, des, action, log_pi, next_state, device):
    # Prepare tensors
    state_tensor = torch.tensor([state], dtype=torch.long).to(device)
    des_tensor = torch.tensor([des], dtype=torch.long).to(device)
    act_tensor = torch.tensor([action], dtype=torch.long).to(device)
    next_state_tensor = torch.tensor([next_state], dtype=torch.long).to(device)
    log_pi_tensor = torch.tensor([log_pi], dtype=torch.float).to(device)

    # Calculate reward
    with torch.no_grad():
        reward = discriminator_net.calculate_reward(
            state_tensor, des_tensor, act_tensor, log_pi_tensor, next_state_tensor
        )
    return reward.item()

for index, row in df.iterrows():
    learner_trajectory_str = str(row['Learner Trajectory'])
    origin = int(row['Origin'])
    destination = int(row['Destination'])
    diff_from_index = row['Different_From_Index']
    same_again_index = row['Same_Again_From_Index']

    # Skip rows with missing trajectory
    if pd.isnull(learner_trajectory_str) or learner_trajectory_str.strip() == '':
        continue

    # Process the Learner Trajectory
    learner_traj = learner_trajectory_str.strip().split('_')
    learner_traj = [int(node) for node in learner_traj]

    # Initialize lists to store rewards and actions
    rewards_list = []
    actions_list = []
    log_pis_list = []
    states_list = learner_traj[:-1]  # All states except the last one
    destination_node = learner_traj[-1]  # Assuming the destination is the last node
    next_states_list = learner_traj[1:]  # States shifted by one

    # Loop over the states in the trajectory
    for idx, state in enumerate(states_list):
        current_state = state
        next_state = next_states_list[idx]

        # Get CNN input for policy network
        input_data = get_cnn_input(policy_net, current_state, destination_node, device)
        input_data.requires_grad = False

        # Prepare tensors
        state_tensor = torch.tensor([current_state], dtype=torch.long).to(device)
        des_tensor = torch.tensor([destination_node], dtype=torch.long).to(device)

        # Get action probabilities from policy network
        with torch.no_grad():
            output = policy_net.forward(input_data)
            x_mask = policy_net.policy_mask[state_tensor]
            output = output.masked_fill((1 - x_mask).bool(), -1e32)
            action_probs = torch.softmax(output, dim=1)
            # Find the action that leads to the next_state
            possible_actions = env.state_action[state]
            action_indices = np.where(possible_actions == next_state)[0]
            if len(action_indices) > 0:
                action = action_indices[0]
                log_pi = torch.log(action_probs[0, action] + 1e-8)
            else:
                # Next state is not reachable from current state
                action = None
                log_pi = 0.0

        actions_list.append(action)
        log_pis_list.append(log_pi.item())

        # Compute reward
        if action is not None:
            reward = compute_reward(
                discriminator_net, current_state, destination_node, action, log_pi.item(), next_state, device
            )
        else:
            reward = 0.0  # Assign zero reward if action is invalid
        rewards_list.append(reward)

    # Convert rewards_list to numpy array for easier indexing
    rewards_array = np.array(rewards_list)

    # Find highest and lowest reward indices and values
    highest_reward_index = np.argmax(rewards_array)
    highest_reward_value = rewards_array[highest_reward_index]
    lowest_reward_index = np.argmin(rewards_array)
    lowest_reward_value = rewards_array[lowest_reward_index]

    # Get rewards at Different_From_Index and Same_Again_From_Index
    diff_index_reward = None
    same_again_index_reward = None

    # Ensure indices are valid integers
    if pd.notnull(diff_from_index):
        diff_from_index = int(diff_from_index)
        if diff_from_index >= 0 and diff_from_index < len(rewards_array):
            diff_index_reward = rewards_array[diff_from_index]
    if pd.notnull(same_again_index):
        same_again_index = int(same_again_index)
        if same_again_index >= 0 and same_again_index < len(rewards_array):
            same_again_index_reward = rewards_array[same_again_index]

    # Record the results in the DataFrame
    df.loc[index, 'Highest_Reward_Index'] = highest_reward_index+1
    df.loc[index, 'Highest_Reward_Value'] = highest_reward_value
    df.loc[index, 'Lowest_Reward_Index'] = lowest_reward_index+1
    df.loc[index, 'Lowest_Reward_Value'] = lowest_reward_value
    df.loc[index, 'Diff_Index_Reward'] = diff_index_reward
    df.loc[index, 'Same_Again_Index_Reward'] = same_again_index_reward

df.to_excel('same_OD_different_paths_rewards.xlsx', index=False)
