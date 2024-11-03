import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import pandas as pd
import os

# Captum imports for interpretability
from captum.attr import IntegratedGradients, FeatureAblation

# Custom modules (adjust the import paths as necessary)
from network_env import RoadWorld
from utils.load_data import (
    load_test_traj,
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

    policy_net.load_state_dict(torch.load(model_path, map_location=device))

    policy_net.eval()
    policy_net.to_device(device)

    return policy_net

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

    # Path settings (adjust paths as necessary)
    cv = 0  # Cross-validation index
    size = 10000  # Size of the training data
    model_path = f"../trained_models/base/bc_CV{cv}_size{size}.pt"
    edge_p = "../data/base/edge.txt"
    network_p = "../data/base/transit.npy"
    path_feature_p = "../data/base/feature_od.npy"
    test_p = f"../data/base/cross_validation/test_CV{cv}.csv"

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
    policy_net = load_model(
        model_path, device, env, path_feature_pad, edge_feature_pad
    )

    # Load test trajectories
    test_trajs, test_od = load_test_traj(test_p)

    # Ensure output directories exist
    output_dir = 'output_img_bc'
    os.makedirs(output_dir, exist_ok=True)

    # Initialize variables to accumulate attributions and importance scores
    total_channel_importance_ig = None
    total_channel_importance_ablation = None
    total_steps = 0

    # Loop over all trajectories
    for traj_idx, traj in enumerate(test_trajs):
        states_list = [int(s) for s in traj[:-1]]  # All states except the last one
        destination = int(traj[-1])  # The destination is the last state

        # Initialize lists to store attributions and importance scores for the trajectory
        channel_importance_ig_list = []
        channel_importance_ablation_list = []

        # Loop over the states in the trajectory
        for idx, state in enumerate(states_list):
            # Get CNN input for policy network
            input_data = get_cnn_input(policy_net, state, destination, device)
            input_data.requires_grad = True

            # Prepare tensors
            state_tensor = torch.tensor([state], dtype=torch.long).to(device)
            des_tensor = torch.tensor([destination], dtype=torch.long).to(device)

            # Get action probabilities from policy network
            with torch.no_grad():
                output = policy_net.forward(input_data)
                x_mask = policy_net.policy_mask[state_tensor]
                output = output.masked_fill((1 - x_mask).bool(), -1e32)
                action_probs = torch.softmax(output, dim=1)
                predicted_action = torch.argmax(action_probs, dim=1)

            # Compute attributions for policy network
            def forward_func_policy(input_data):
                x = policy_net.forward(input_data)
                x_mask = policy_net.policy_mask[state_tensor]
                x = x.masked_fill((1 - x_mask).bool(), -1e32)
                action_probs = torch.softmax(x, dim=1)
                return action_probs

            # Integrated Gradients for policy network
            ig = IntegratedGradients(forward_func_policy)
            attributions_ig = ig.attribute(input_data, target=predicted_action)
            attributions_ig_np = attributions_ig.squeeze().cpu().detach().numpy()

            # Feature Ablation for policy network
            feature_ablation = FeatureAblation(forward_func_policy)
            attributions_ablation = feature_ablation.attribute(input_data, target=predicted_action)
            attributions_ablation_np = attributions_ablation.squeeze().cpu().detach().numpy()

            # Aggregate attributions for policy network
            channel_importance_ig = np.sum(np.abs(attributions_ig_np), axis=(1, 2))
            channel_importance_ablation = np.sum(np.abs(attributions_ablation_np), axis=(1, 2))
            channel_importance_ig_list.append(channel_importance_ig)
            channel_importance_ablation_list.append(channel_importance_ablation)

            total_steps += 1

        # Sum importance scores over the trajectory
        traj_channel_importance_ig = np.sum(channel_importance_ig_list, axis=0)
        traj_channel_importance_ablation = np.sum(channel_importance_ablation_list, axis=0)

        # Accumulate importance scores over all trajectories
        if total_channel_importance_ig is None:
            total_channel_importance_ig = traj_channel_importance_ig
            total_channel_importance_ablation = traj_channel_importance_ablation
        else:
            total_channel_importance_ig += traj_channel_importance_ig
            total_channel_importance_ablation += traj_channel_importance_ablation

    # Compute average importance scores
    avg_channel_importance_ig = total_channel_importance_ig / total_steps
    avg_channel_importance_ablation = total_channel_importance_ablation / total_steps

    # Process and visualize the aggregated feature importance
    # Policy Network - Integrated Gradients
    ranked_indices = np.argsort(-avg_channel_importance_ig)
    sorted_features = [feature_names[i] for i in ranked_indices]
    sorted_importance = avg_channel_importance_ig[ranked_indices]

    # Create DataFrame
    feature_importance_df = pd.DataFrame({
        'Feature': sorted_features,
        'Importance Score': sorted_importance
    })

    # Save DataFrame to CSV
    feature_importance_df.to_csv(
        os.path.join(output_dir, 'policy_ig_feature_importance_all_steps.csv'), index=False
    )

    # Plot Feature Importance
    plt.figure(figsize=(10, 6))
    plt.barh(sorted_features[::-1], sorted_importance[::-1])
    plt.xlabel('Average Importance Score')
    plt.title('Policy Network IG Feature Importance - All Steps (BC Policy)')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'policy_ig_feature_importance_all_steps.png'))
    plt.close()

    # Policy Network - Feature Ablation
    ranked_indices_ablation = np.argsort(-avg_channel_importance_ablation)
    sorted_features_ablation = [feature_names[i] for i in ranked_indices_ablation]
    sorted_importance_ablation = avg_channel_importance_ablation[ranked_indices_ablation]

    # Create DataFrame
    feature_importance_ablation_df = pd.DataFrame({
        'Feature': sorted_features_ablation,
        'Importance Score': sorted_importance_ablation
    })

    # Save DataFrame to CSV
    feature_importance_ablation_df.to_csv(
        os.path.join(output_dir, 'policy_ablation_feature_importance_all_steps.csv'), index=False
    )

    # Plot Feature Importance
    plt.figure(figsize=(10, 6))
    plt.barh(sorted_features_ablation[::-1], sorted_importance_ablation[::-1])
    plt.xlabel('Average Importance Score')
    plt.title('Policy Network Feature Ablation - All Steps (BC Policy)')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'policy_ablation_feature_importance_all_steps.png'))
    plt.close()

    print("Interpretation complete. Results saved in the 'output_img_bc' directory.")

if __name__ == "__main__":
    interpret_model()