import torch
import torch.nn.functional as F
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os

# Captum imports for interpretability
from captum.attr import IntegratedGradients, visualization as viz

# Custom modules (adjust the import paths as necessary)
from network_env import RoadWorld
from utils.load_data import (
    ini_od_dist,
    load_path_feature,
    load_link_feature,
    minmax_normalization,
)
from model.policy import PolicyCNN

def load_model(model_path, device, env, path_feature_pad, edge_feature_pad):
    policy_net = PolicyCNN(
        env.n_actions,
        env.policy_mask,
        env.state_action,
        path_feature_pad,
        edge_feature_pad,
        path_feature_pad.shape[-1] + edge_feature_pad.shape[-1] + 1,
        env.pad_idx,
    ).to(device)

    model_dict = torch.load(model_path, map_location=device)
    policy_net.load_state_dict(model_dict['Policy'])
    policy_net.eval()
    return policy_net

def get_cnn_input(policy_net, state, des, device):
    state = torch.tensor([state], dtype=torch.long).to(device)
    des = torch.tensor([des], dtype=torch.long).to(device)
    # Process features to get the CNN input
    input_data = policy_net.process_features(state, des)
    return input_data, state

def interpret_model():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

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
    policy_net = load_model(
        model_path, device, env, path_feature_pad, edge_feature_pad
    )
    policy_net.to(device)

    # Read generated trajectories
    df_generated = pd.read_csv(generated_trajs_csv)

    # Prepare data for Captum
    input_data_list = []
    state_list = []
    des_list = []
    action_list = []

    for idx, row in df_generated.iterrows():
        trajectory_str = row['generated_trajectory']
        actions_taken = eval(row['actions_taken'])
        destination_node = int(row['destination'])

        trajectory = [int(s) for s in trajectory_str.strip().split('_')]
        states = trajectory[:-1]

        # Ensure actions_taken matches states
        if len(states) != len(actions_taken):
            continue  # Skip if data is inconsistent

        for state, action in zip(states, actions_taken):
            # Prepare CNN input data
            input_data, state_tensor = get_cnn_input(policy_net, state, destination_node, device)
            input_data_list.append(input_data)
            state_list.append(state_tensor)
            des_list.append(torch.tensor([destination_node], dtype=torch.long).to(device))
            action_list.append(action)

    # Convert lists to tensors
    inputs = torch.cat(input_data_list, dim=0).to(device)  # Shape: [num_samples, channels, height, width]
    states = torch.cat(state_list, dim=0).to(device)       # Shape: [num_samples]
    des = torch.cat(des_list, dim=0).to(device)            # Shape: [num_samples]
    actions = torch.tensor(action_list, dtype=torch.long).to(device)  # Shape: [num_samples]

    # Select a subset for interpretation
    num_samples = 10  # Adjust as needed
    inputs = inputs[:num_samples]
    states = states[:num_samples]
    des = des[:num_samples]
    actions = actions[:num_samples]

    # Define the forward function for Captum
    def forward_func(input_data):
        output = policy_net.forward(input_data)
        x_mask = policy_net.policy_mask[states]
        output = output.masked_fill((1 - x_mask).bool(), -1e32)
        action_probs = F.softmax(output, dim=1)
        # We can select the probabilities corresponding to the taken actions
        selected_action_probs = action_probs.gather(1, actions.unsqueeze(1)).squeeze(1)
        return selected_action_probs

    # Initialize Integrated Gradients
    ig = IntegratedGradients(forward_func)

    # Compute attributions
    attributions_ig = ig.attribute(inputs, target=None, n_steps=50)

    # Aggregate attributions over the channels
    attributions_ig_sum = attributions_ig.sum(dim=2).sum(dim=2)  # Sum over height and width

    # Map attributions to feature names
    attributions_per_feature = attributions_ig_sum.cpu().detach().numpy()

    # Create DataFrame for attributions
    for i in range(num_samples):
        feature_attributions = attributions_per_feature[i]
        feature_importance = dict(zip(feature_names, feature_attributions))
        sorted_features = sorted(feature_importance.items(), key=lambda x: abs(x[1]), reverse=True)
        print(f"\nSample {i+1} Feature Importances:")
        for feature, importance in sorted_features:
            print(f"{feature}: {importance}")

        # Optionally, visualize the attributions
        input_np = inputs[i].cpu().detach().numpy()
        attribution_np = attributions_ig[i].cpu().detach().numpy()

        # Visualization using Captum's utility
        viz.visualize_image_attr_multiple(
            np.transpose(attribution_np, (1, 2, 0)),
            np.transpose(input_np, (1, 2, 0)),
            methods=["heat_map"],
            show_colorbar=True,
            titles=["Attributions"],
            outlier_perc=2,
        )

    print("Interpretation complete.")

if __name__ == "__main__":
    interpret_model()
