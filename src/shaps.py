from utils.evaluation import evaluate_model, evaluate_log_prob, evaluate_train_edit_dist
import time
import torch
from utils.load_data import ini_od_dist, load_path_feature, load_link_feature, \
    minmax_normalization, load_train_sample, load_test_traj
from network_env import RoadWorld
from utils.torch import to_device
import numpy as np
import pandas as pd
from model.policy import PolicyCNN
from model.value import ValueCNN
from model.discriminator import DiscriminatorAIRLCNN
import csv
import shap
import matplotlib.pyplot as plt
import torch.nn.functional as F

def load_model(model_path, device, env, path_feature_pad, edge_feature_pad):
    # Assuming the dimensions for the models based on training setup
    gamma = 0.95  # discount factor
    edge_data = pd.read_csv('../data/base/edge.txt')
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

    model_dict = torch.load(model_path, map_location=device)
    policy_net.load_state_dict(model_dict['Policy'])
    value_net.load_state_dict(model_dict['Value'])
    discrim_net.load_state_dict(model_dict['Discrim'])

    return policy_net, value_net, discrim_net

device = torch.device('cpu')
    
# Path settings
# model_path = "../trained_models/base/airl_CV0_size10000.pt"  # Adjust as necessary
model_path = "../trained_models/base/bleu90.pt" #bleu90.pt 9nzx7df7
edge_p = "../data/base/edge.txt"
network_p = "../data/base/transit.npy"
path_feature_p = "../data/base/feature_od.npy"
train_p = "../data/base/cross_validation/train_CV0_size10000.csv"
test_p = "../data/base/cross_validation/test_CV0.csv"

# Initialize environment
od_list, od_dist = ini_od_dist(train_p)
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


# Assuming policy_net and discrim_net are your policy and reward networks
policy_net.eval()
discrim_net.eval()


# Create some sample data
num_samples = 100

# Paths to data
generated_trajs_csv = "./eva/generated_trajectories_with_rewards.csv"

# # Load the models
# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# policy_net, value_net, discriminator_net = load_model(
#     model_path, device, env, path_feature_pad, edge_feature_pad
# )

# Read generated trajectories
df_generated = pd.read_csv(generated_trajs_csv)

# Prepare data for SHAP
input_data_list = []
output_data_list = []

for idx, row in df_generated.iterrows():
    trajectory_str = row['generated_trajectory']
    actions_taken = eval(row['actions_taken'])
    destination_node = int(row['destination'])

    trajectory = [int(s) for s in trajectory_str.strip().split('_')]
    states = trajectory[:-1]
    next_states = trajectory[1:]

    # Ensure actions_taken matches states
    if len(states) != len(actions_taken):
        continue  # Skip if data is inconsistent

    for state, action, next_state in zip(states, actions_taken, next_states):
        # Prepare inputs
        input_data = {
            'state': state,
            'des': destination_node,
            'action': action,
            'next_state': next_state
        }
        input_data_list.append(input_data)

        # Prepare outputs (e.g., discriminator output)
        with torch.no_grad():
            state_tensor = torch.tensor([state], dtype=torch.long).to(device)
            des_tensor = torch.tensor([destination_node], dtype=torch.long).to(device)
            action_tensor = torch.tensor([action], dtype=torch.long).to(device)
            next_state_tensor = torch.tensor([next_state], dtype=torch.long).to(device)

            log_pis = torch.zeros_like(action_tensor, dtype=torch.float32).to(device)
            output = discrim_net.forward(state_tensor, des_tensor, action_tensor, log_pis, next_state_tensor)
            output_data_list.append(output.cpu().numpy().flatten())

# Convert input_data_list to numpy array
def convert_input_data(input_data_list):
    input_array = []
    for data in input_data_list:
        input_array.append([data['state'], data['des'], data['action'], data['next_state']])
    return np.array(input_array)

input_array = convert_input_data(input_data_list)
output_array = np.array(output_data_list)

# Define prediction functions
def policy_predict(input_array):
    # Input array shape: [num_samples, 2]
    num_samples = input_array.shape[0]
    outputs = []
    for i in range(num_samples):
        state = torch.tensor([int(input_array[i, 0])], dtype=torch.long).to(device)
        des = torch.tensor([int(input_array[i, 1])], dtype=torch.long).to(device)

        with torch.no_grad():
            x = policy_net.process_features(state, des)
            output = policy_net.forward(x)
            action_probs = F.softmax(output, dim=1)
            outputs.append(action_probs.cpu().numpy().flatten())
    return np.array(outputs)

def discriminator_predict(input_array):
    # Input array shape: [num_samples, 4]
    num_samples = input_array.shape[0]
    outputs = []
    for i in range(num_samples):
        state = torch.tensor([int(input_array[i, 0])], dtype=torch.long).to(device)
        des = torch.tensor([int(input_array[i, 1])], dtype=torch.long).to(device)
        action = torch.tensor([int(input_array[i, 2])], dtype=torch.long).to(device)
        next_state = torch.tensor([int(input_array[i, 3])], dtype=torch.long).to(device)

        log_pis = torch.zeros_like(action, dtype=torch.float32).to(device)
        with torch.no_grad():
            output = discrim_net.forward(state, des, action, log_pis, next_state)
            outputs.append(output.cpu().numpy().flatten())
    return np.array(outputs)

# Create background dataset
policy_background = input_array[:100, :2]
discriminator_background = input_array[:100]

# Create the SHAP explainers
policy_explainer = shap.KernelExplainer(policy_predict, policy_background)
discriminator_explainer = shap.KernelExplainer(discriminator_predict, discriminator_background)

# Select test samples
policy_test_samples = input_array[100:110, :2]
discriminator_test_samples = input_array[100:110]

# Compute SHAP values
policy_shap_values = policy_explainer.shap_values(policy_test_samples)
discriminator_shap_values = discriminator_explainer.shap_values(discriminator_test_samples)

# Visualize SHAP values
# For the policy network, since the output is probabilities over actions, shap_values will be a list
# You can choose an action index to visualize
action_index = 0  # Change as needed

# Visualize policy SHAP values for the selected action
shap.summary_plot(policy_shap_values[action_index], policy_test_samples, feature_names=['State', 'Destination'])

# Visualize discriminator SHAP values
shap.summary_plot(discriminator_shap_values, discriminator_test_samples, feature_names=['State', 'Destination', 'Action', 'Next State'])