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


# def aggregate_and_rank_shap_values(expanded_feature_names, shap_values_action):
#     """
#     Aggregate SHAP values for each feature type (across grid cells) and rank them.
#     """
#     # Ensure SHAP values and feature names align
#     num_features = shap_values_action.shape[1]
#     feature_names_truncated = expanded_feature_names[:num_features]

#     aggregated_shap_values = {}

#     # Iterate through unique feature names (excluding grid coordinates like `_0_1`)
#     for base_feature in set([name.split("_")[0] for name in feature_names_truncated]):
#         # Get all feature names matching the base feature (e.g., 'Total Length')
#         matching_features = [name for name in feature_names_truncated if name.startswith(base_feature)]

#         # Find the indices of these features
#         indices = [feature_names_truncated.index(name) for name in matching_features]

#         # Sum SHAP values across all matching indices (absolute values for importance ranking)
#         aggregated_shap_values[base_feature] = np.sum(np.abs(shap_values_action[:, indices]))

#     # Create a DataFrame for ranking
#     aggregated_df = pd.DataFrame({
#         'Feature': aggregated_shap_values.keys(),
#         'SHAP Value': aggregated_shap_values.values()
#     }).sort_values(by='SHAP Value', ascending=False)

#     return aggregated_df

def aggregate_and_rank_shap_values(expanded_feature_names, shap_values_action):
    """
    Aggregate SHAP values for each feature type (across grid cells) and rank them.
    """
    # Ensure SHAP values and feature names align
    num_features = shap_values_action.shape[1]
    feature_names_truncated = expanded_feature_names[:num_features]

    aggregated_shap_values = {}

    # Iterate through unique feature names (excluding grid coordinates like `_0_1`)
    for base_feature in set([name.split("_")[0] for name in feature_names_truncated]):
        # Get all feature names matching the base feature (e.g., 'Total Length')
        matching_features = [name for name in feature_names_truncated if name.startswith(base_feature)]

        # Find the indices of these features
        indices = [feature_names_truncated.index(name) for name in matching_features]

        # Sum SHAP values across all matching indices (absolute values for importance ranking)
        aggregated_shap_values[base_feature] = np.sum(np.abs(shap_values_action[:, indices]), axis=1).mean()

    # Create a DataFrame for ranking
    aggregated_df = pd.DataFrame({
        'Feature': aggregated_shap_values.keys(),
        'SHAP Value': aggregated_shap_values.values()
    }).sort_values(by='SHAP Value', ascending=False)

    return aggregated_df



def plot_aggregated_shap(aggregated_df):
    """
    Create a horizontal bar plot for aggregated SHAP values.
    """
    plt.figure(figsize=(10, 6))
    plt.barh(aggregated_df['Feature'], aggregated_df['SHAP Value'], color='skyblue')
    plt.xlabel('Aggregated SHAP Value (Importance)')
    plt.title('Feature Importance (Aggregated Across Cells)')
    plt.gca().invert_yaxis()  # Largest importance at the top
    plt.tight_layout()
    plt.show()


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
    state_list = []

    for idx, row in df_generated.iterrows():
        trajectory_str = row['generated_trajectory']
        actions_taken = eval(row['actions_taken'])
        destination_node = int(row['destination'])

        trajectory = [int(s) for s in trajectory_str.strip().split('_')]
        states = trajectory[:-1]

        for state in states:
            # Prepare CNN input data
            input_data = get_cnn_input(policy_net, state, destination_node, device)
            input_data_np = input_data.cpu().detach().numpy().squeeze(0)
            input_data_list.append(input_data_np)
            state_list.append(state)

    # Convert input_data_list to numpy array
    input_array = np.array(input_data_list)  # Shape: [num_samples, channels, height, width]

    # Flatten input data for SHAP
    num_samples = input_array.shape[0]
    num_channels = input_array.shape[1]
    height = input_array.shape[2]
    width = input_array.shape[3]

    # Reshape input data to [num_samples, num_features]
    input_array_flat = input_array.reshape(num_samples, -1)

    # Expand feature names to match flattened input
    expanded_feature_names = [
        f"{feature_names[c]}_{h}_{w}" for c in range(num_channels) for h in range(height) for w in range(width)
    ]

    # Define the prediction function for the policy network
    def policy_predict(input_array_flat):
        outputs = []
        for i in range(input_array_flat.shape[0]):
            input_data_np = input_array_flat[i].reshape(num_channels, height, width)
            input_data_tensor = torch.tensor(input_data_np, dtype=torch.float32).unsqueeze(0).to(device)
            with torch.no_grad():
                output = policy_net(input_data_tensor)
                outputs.append(output.cpu().numpy().flatten())
        return np.array(outputs)

    # Create background dataset
    policy_background = shap.kmeans(input_array_flat, 50)

    # Create the SHAP explainer
    policy_explainer = shap.KernelExplainer(policy_predict, policy_background)

    # Select test samples
    policy_test_samples = input_array_flat[:30]

    # Compute SHAP values
    policy_shap_values = policy_explainer.shap_values(policy_test_samples)

    # Aggregate SHAP values and rank features
    aggregated_df = aggregate_and_rank_shap_values(expanded_feature_names, policy_shap_values[0])

    # Plot the aggregated SHAP values
    plot_aggregated_shap(aggregated_df)


if __name__ == "__main__":
    interpret_model()
