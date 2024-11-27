import torch
import torch.nn.functional as F
import numpy as np
import pandas as pd
import shap
import matplotlib.pyplot as plt
import os

# Custom modules (adjust the import paths as necessary)
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

def get_cnn_input(policy_net, state, des, device):
    state = torch.tensor([state], dtype=torch.long).to(device)
    des = torch.tensor([des], dtype=torch.long).to(device)
    # Process features to get the CNN input
    input_data = policy_net.process_features(state, des)
    return input_data

def interpret_model():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Feature names (ensure this matches the number of channels in your input data)
    feature_names = [
        # Path features (12 features)
        'Number of links',                # 0
        'Total length',                   # 1
        'Number of left turns',           # 2
        'Number of right turns',          # 3
        'Number of U-turns',              # 4
        'Number of residential roads',    # 5
        'Number of primary roads',        # 6
        'Number of unclassified roads',   # 7
        'Number of tertiary roads',       # 8
        'Number of living_street roads',  # 9
        'Number of secondary roads',      #10
        'Mask feature',                   #11
        # Edge features (8 features)
        'Edge length',                    #12
        'Highway type: residential',      #13
        'Highway type: primary',          #14
        'Highway type: unclassified',     #15
        'Highway type: tertiary',         #16
        'Highway type: living_street',    #17
        'Highway type: secondary',        #18
        'Neighbor mask'                   #19
    ]

    # Paths to data and models (adjust paths as necessary)
    data_p = "../data/base/cross_validation/train_CV0_size10000.csv"
    model_path = "../trained_models/base/bleu90.pt"  # Adjust as necessary
    edge_p = "../data/base/edge.txt"
    network_p = "../data/base/transit.npy"
    path_feature_p = "../data/base/feature_od.npy"
    generated_trajs_csv = "./eva/generated_trajectories_with_rewards.csv"

    # Extract the model name from the model_path
    model_name = os.path.splitext(os.path.basename(model_path))[0]

    # Initialize environment
    od_list, od_dist = ini_od_dist(data_p)
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
    policy_net, value_net, discriminator_net = load_model(
        model_path, device, env, path_feature_pad, edge_feature_pad
    )

    # Read generated trajectories
    df_generated = pd.read_csv(generated_trajs_csv)

    # Prepare data for SHAP
    input_data_list = []
    output_data_list = []
    state_list = []

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
            # Prepare CNN input data
            input_data = get_cnn_input(policy_net, state, destination_node, device)
            input_data_np = input_data.cpu().detach().numpy().squeeze(0)
            input_data_flat = input_data_np.flatten()

            # Include state as an additional feature
            input_data_with_state = np.concatenate(([state], input_data_flat))
            input_data_list.append(input_data_with_state)

            # Store the action probabilities
            with torch.no_grad():
                output = policy_net.forward(input_data)
                x_mask = policy_net.policy_mask[torch.tensor([state], dtype=torch.long)]
                output = output.masked_fill((1 - x_mask).bool(), -1e32)
                action_probs = torch.softmax(output, dim=1)
                output_data_list.append(action_probs.cpu().numpy().flatten())

    # Convert input_data_list to numpy array
    input_array = np.array(input_data_list)  # Shape: [num_samples, 1 + num_features]
    output_array = np.array(output_data_list)

    # Extract dimensions
    num_samples = input_array.shape[0]
    total_features = input_array.shape[1]  # Includes state
    num_channels = input_data_np.shape[0]
    height = input_data_np.shape[1]
    width = input_data_np.shape[2]
    num_features = total_features - 1  # Exclude state

    # Adjust feature names
    expanded_feature_names = ['state']
    for c in range(num_channels):
        for h in range(height):
            for w in range(width):
                expanded_feature_names.append(f"{feature_names[c]}_{h}_{w}")

    # Define the prediction function for the policy network
    def policy_predict(input_array_with_state):
        # Input array shape: [num_samples, 1 + num_features]
        num_samples = input_array_with_state.shape[0]
        outputs = []
        for i in range(num_samples):
            input_data_with_state = input_array_with_state[i]
            state = int(input_data_with_state[0])
            input_data_flat = input_data_with_state[1:]
            input_data_np = input_data_flat.reshape(num_channels, height, width)
            input_data_tensor = torch.tensor(input_data_np, dtype=torch.float32).unsqueeze(0).to(device)
            with torch.no_grad():
                output = policy_net.forward(input_data_tensor)
                x_mask = policy_net.policy_mask[torch.tensor([state], dtype=torch.long)]
                output = output.masked_fill((1 - x_mask).bool(), -1e32)
                action_probs = torch.softmax(output, dim=1)
                outputs.append(action_probs.cpu().numpy().flatten())
        return np.array(outputs)

    # Create background dataset
    policy_background = shap.kmeans(input_array, 50)

    # Create the SHAP explainer
    policy_explainer = shap.KernelExplainer(policy_predict, policy_background)

    # Select test samples
    policy_test_samples = input_array[:3]

    # Compute SHAP values
    policy_shap_values = policy_explainer.shap_values(policy_test_samples)

    # Since policy_shap_values is a list of arrays (one per action)
    action_index = 0  # Change as needed
    policy_shap_values_action = policy_shap_values[action_index]

    # Check shapes
    print('policy_shap_values_action shape:', policy_shap_values_action.shape)
    print('policy_test_samples shape:', policy_test_samples.shape)

    # Exclude 'state' feature from policy_test_samples and feature names
    policy_test_samples_no_state = policy_test_samples[:, 1:]  # Exclude the first column (state)
    expanded_feature_names_no_state = expanded_feature_names[1:]

    # Now, policy_shap_values_action and policy_test_samples_no_state should have matching shapes
    print('policy_test_samples_no_state shape:', policy_test_samples_no_state.shape)
    print('Number of features in SHAP values:', policy_shap_values_action.shape[1])
    print('Number of features in test samples:', policy_test_samples_no_state.shape[1])

    # Visualize SHAP values
    shap.summary_plot(policy_shap_values_action, policy_test_samples_no_state, feature_names=expanded_feature_names_no_state)

if __name__ == "__main__":
    interpret_model()
