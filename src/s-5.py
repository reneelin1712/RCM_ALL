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

def get_discriminator_input(discriminator_net, state, des, device):
    state = torch.tensor([state], dtype=torch.long).to(device)
    des = torch.tensor([des], dtype=torch.long).to(device)
    input_data_disc = discriminator_net.process_neigh_features(state, des)
    return input_data_disc

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
    des_list = []
    act_list = []
    log_pi_list = []
    next_state_list = []
    input_data_disc_list = []
    output_data_disc_list = []

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
            # Prepare CNN input data for policy network
            input_data = get_cnn_input(policy_net, state, destination_node, device)
            input_data_np = input_data.cpu().detach().numpy().squeeze(0)
            input_data_list.append(input_data_np)
            state_list.append(state)
            des_list.append(destination_node)
            act_list.append(action)
            next_state_list.append(next_state)

            # Prepare outputs (e.g., model output)
            with torch.no_grad():
                output = policy_net.forward(input_data)
                x_mask = policy_net.policy_mask[torch.tensor([state], dtype=torch.long)]
                output = output.masked_fill((1 - x_mask).bool(), -1e32)
                action_probs = torch.softmax(output, dim=1)
                output_data_list.append(action_probs.cpu().numpy().flatten())

                # Compute log_pi
                log_pi = torch.log(action_probs[0, action] + 1e-8).item()
                log_pi_list.append(log_pi)

            # Prepare input data for discriminator network
            input_data_disc = get_discriminator_input(discriminator_net, state, destination_node, device)
            input_data_disc_np = input_data_disc.cpu().detach().numpy().squeeze(0)
            input_data_disc_list.append(input_data_disc_np)

            # Prepare discriminator output
            state_tensor = torch.tensor([state], dtype=torch.long).to(device)
            des_tensor = torch.tensor([destination_node], dtype=torch.long).to(device)
            act_tensor = torch.tensor([action], dtype=torch.long).to(device)
            log_pi_tensor = torch.tensor([log_pi], dtype=torch.float32).to(device)
            next_state_tensor = torch.tensor([next_state], dtype=torch.long).to(device)

            with torch.no_grad():
                discriminator_output = discriminator_net.forward(
                    state_tensor, des_tensor, act_tensor, log_pi_tensor, next_state_tensor
                )
                output_data_disc_list.append(discriminator_output.cpu().numpy().flatten())

    # Convert input_data_list to numpy array
    input_array = np.array(input_data_list)  # Shape: [num_samples, channels, height, width]
    output_array = np.array(output_data_list)
    input_array_disc = np.array(input_data_disc_list)
    output_array_disc = np.array(output_data_disc_list)

    # Prepare input tensors for SHAP (Policy Network)
    num_samples = input_array.shape[0]
    num_channels = input_array.shape[1]
    height = input_array.shape[2]
    width = input_array.shape[3]

    # Reshape input data to [num_samples, num_features]
    input_array_flat = input_array.reshape(num_samples, -1)
    num_features = input_array_flat.shape[1]

    # Expand feature names to match flattened input
    expanded_feature_names = []
    for c in range(num_channels):
        for h in range(height):
            for w in range(width):
                expanded_feature_names.append(f"{feature_names[c]}_{h}_{w}")

    # Define the prediction function for the policy network
    def policy_predict(input_array_flat):
        # Input array shape: [num_samples, num_features]
        num_samples = input_array_flat.shape[0]
        outputs = []
        for i in range(num_samples):
            # Reconstruct input_data tensor
            input_data_np = input_array_flat[i].reshape(num_channels, height, width)
            input_data_tensor = torch.tensor(input_data_np, dtype=torch.float32).unsqueeze(0).to(device)
            with torch.no_grad():
                output = policy_net.forward(input_data_tensor)
                x_mask = policy_net.policy_mask[torch.tensor([state_list[i]], dtype=torch.long)]
                output = output.masked_fill((1 - x_mask).bool(), -1e32)
                action_probs = torch.softmax(output, dim=1)
                outputs.append(action_probs.cpu().numpy().flatten())
        return np.array(outputs)

    # Create background dataset for policy network
    policy_background = shap.kmeans(input_array_flat, 50)

    # Create the SHAP explainer for policy network
    policy_explainer = shap.KernelExplainer(policy_predict, policy_background)

    # Select test samples for policy network
    policy_test_samples = input_array_flat[:30]

    # Compute SHAP values for policy network
    policy_shap_values = policy_explainer.shap_values(policy_test_samples)

    # Visualize SHAP values for policy network
    action_index = 0  # Change as needed

    if isinstance(policy_shap_values, list):
        # SHAP values is a list
        policy_shap_values_action = policy_shap_values[action_index]
    else:
        # SHAP values is an array
        policy_shap_values_action = policy_shap_values[:, :, action_index]

    # Plot summary plot for policy network
    shap.summary_plot(policy_shap_values_action, policy_test_samples, feature_names=expanded_feature_names)

    # Prepare inputs for discriminator network
    state_tensor_full = torch.tensor(state_list, dtype=torch.long).to(device)
    des_tensor_full = torch.tensor(des_list, dtype=torch.long).to(device)
    act_tensor_full = torch.tensor(act_list, dtype=torch.long).to(device)
    log_pi_tensor_full = torch.tensor(log_pi_list, dtype=torch.float32).to(device)
    next_state_tensor_full = torch.tensor(next_state_list, dtype=torch.long).to(device)
    input_data_disc_full = torch.tensor(input_array_disc, dtype=torch.float32).to(device)

    # Select background data and test samples for discriminator network
    background_size = 50
    test_size = 30

    background_indices = np.random.choice(len(state_list), size=background_size, replace=False)
    test_indices = np.random.choice(len(state_list), size=test_size, replace=False)

    background_inputs = [
        input_data_disc_full[background_indices],
        act_tensor_full[background_indices],
        log_pi_tensor_full[background_indices],
        next_state_tensor_full[background_indices],
    ]

    test_inputs = [
        input_data_disc_full[test_indices],
        act_tensor_full[test_indices],
        log_pi_tensor_full[test_indices],
        next_state_tensor_full[test_indices],
    ]

    # Define the wrapper model for discriminator network
    class DiscriminatorModelWrapper(torch.nn.Module):
        def __init__(self, discriminator_net, state_tensor, des_tensor):
            super(DiscriminatorModelWrapper, self).__init__()
            self.discriminator_net = discriminator_net
            self.state_tensor = state_tensor
            self.des_tensor = des_tensor

        def forward(self, input_data_disc, act_tensor, log_pi_tensor, next_state_tensor):
            # Note: state_tensor and des_tensor are fixed
            return self.discriminator_net.forward(
                self.state_tensor, self.des_tensor, act_tensor, log_pi_tensor, next_state_tensor
            )

    # For simplicity, we fix state_tensor and des_tensor to the first sample
    fixed_state_tensor = state_tensor_full[test_indices]
    fixed_des_tensor = des_tensor_full[test_indices]

    discriminator_model_wrapper = DiscriminatorModelWrapper(
        discriminator_net, fixed_state_tensor, fixed_des_tensor
    )

    # Create the SHAP Gradient Explainer for discriminator network
    discriminator_explainer = shap.GradientExplainer(
        discriminator_model_wrapper, background_inputs
    )

    # Compute SHAP values for discriminator network
    discriminator_shap_values = discriminator_explainer.shap_values(test_inputs)

    # Since we're interested in the attributions for input_data_disc
    shap_values_disc_input = discriminator_shap_values[0]  # Index 0 corresponds to input_data_disc

    # Flatten input_data_disc for visualization
    num_samples_disc = test_inputs[0].shape[0]
    num_channels_disc = test_inputs[0].shape[1]
    height_disc = test_inputs[0].shape[2]
    width_disc = test_inputs[0].shape[3]

    test_inputs_flat = test_inputs[0].cpu().detach().numpy().reshape(num_samples_disc, -1)
    shap_values_disc_input_flat = shap_values_disc_input.reshape(num_samples_disc, -1)

    # Visualize SHAP values for discriminator network
    shap.summary_plot(shap_values_disc_input_flat, test_inputs_flat, feature_names=expanded_feature_names)

if __name__ == "__main__":
    interpret_model()
