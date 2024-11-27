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
    state_tensor = torch.tensor([state], dtype=torch.long).to(device)
    des_tensor = torch.tensor([des], dtype=torch.long).to(device)
    # Process features to get the CNN input
    input_data = policy_net.process_features(state_tensor, des_tensor)
    return input_data

def interpret_model():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Feature names (ensure this matches the number of channels in your input data)
    feature_names = [
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
    valid_actions_list = []
    action_taken_list = []

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

        for state, action_taken in zip(states, actions_taken):
            # Prepare CNN input data
            input_data = get_cnn_input(policy_net, state, destination_node, device)
            input_data_np = input_data.cpu().detach().numpy().squeeze(0)
            input_data_list.append(input_data_np)
            state_list.append(state)
            action_taken_list.append(action_taken)

            # Get valid actions for this state
            valid_actions = policy_net.policy_mask[state].nonzero().squeeze(1).cpu().numpy()
            valid_actions_list.append(valid_actions)

    # Convert lists to numpy arrays
    input_array = np.array(input_data_list)
    num_samples = input_array.shape[0]

    # Flatten input data
    input_array_flat = input_array.reshape(num_samples, -1)

    # Expand feature names
    num_channels = input_array.shape[1]
    height = input_array.shape[2]
    width = input_array.shape[3]
    expanded_feature_names = []
    for c in range(num_channels):
        for h in range(height):
            for w in range(width):
                expanded_feature_names.append(f"{feature_names[c]}_{h}_{w}")

    # Define the prediction function for the policy network
    def policy_predict(input_array_flat):
        num_samples = input_array_flat.shape[0]
        outputs = []
        for i in range(num_samples):
            # Reconstruct input_data tensor
            input_data_np = input_array_flat[i].reshape(num_channels, height, width)
            input_data_tensor = torch.tensor(input_data_np, dtype=torch.float32).unsqueeze(0).to(device)
            state = state_list[i]
            with torch.no_grad():
                output = policy_net.forward(input_data_tensor)
                x_mask = policy_net.policy_mask[torch.tensor([state], dtype=torch.long)]
                output = output.masked_fill((1 - x_mask).bool(), -1e32)
                action_probs = torch.softmax(output, dim=1)
                outputs.append(action_probs.cpu().numpy().flatten())
        return np.array(outputs)

    # Create background dataset
    policy_background = shap.kmeans(input_array_flat, 50)

    # Create the SHAP explainer
    policy_explainer = shap.KernelExplainer(policy_predict, policy_background)

    # Select test samples
    test_sample_indices = range(10)  # Adjust as needed
    policy_test_samples = input_array_flat[test_sample_indices]
    action_taken_test = [action_taken_list[i] for i in test_sample_indices]
    valid_actions_test = [valid_actions_list[i] for i in test_sample_indices]

    # Compute SHAP values for each test sample individually
    for i, (test_sample, action_taken, valid_actions) in enumerate(zip(policy_test_samples, action_taken_test, valid_actions_test)):
        # Compute SHAP values
        shap_values = policy_explainer.shap_values(test_sample, nsamples=100)

        # Check if shap_values is a list
        if isinstance(shap_values, list):
            # Since shap_values is a list, we need to find the index of the action taken
            action_index = action_taken
            shap_values_action = shap_values[action_index].reshape(1, -1)
        else:
            # shap_values is an array
            action_index = action_taken
            shap_values_action = shap_values[ :, action_index]

        # Reshape test sample
        test_sample_reshaped = test_sample.reshape(1, -1)

        # Visualize SHAP values
        shap.initjs()
        shap.force_plot(
            policy_explainer.expected_value[action_index],
            shap_values_action,
            test_sample_reshaped,
            feature_names=expanded_feature_names,
            matplotlib=True
        )
        plt.title(f"SHAP Force Plot for Sample {i+1}, Action {action_index}")
        plt.show()

    print("Interpretation complete.")


        # Assuming you have already loaded the discriminator_net and prepared the necessary inputs

    # def discriminator_predict(input_array):
    #     # Input array shape: [num_samples, 4]
    #     num_samples = input_array.shape[0]
    #     outputs = []
    #     for i in range(num_samples):
    #         state = torch.tensor([int(input_array[i, 0])], dtype=torch.long).to(device)
    #         des = torch.tensor([int(input_array[i, 1])], dtype=torch.long).to(device)
    #         action = torch.tensor([int(input_array[i, 2])], dtype=torch.long).to(device)
    #         next_state = torch.tensor([int(input_array[i, 3])], dtype=torch.long).to(device)

    #         log_pis = torch.zeros_like(action, dtype=torch.float32).to(device)
    #         with torch.no_grad():
    #             output = discriminator_net.forward(state, des, action, log_pis, next_state)
    #             outputs.append(output.cpu().numpy().flatten())
    #     return np.array(outputs)

    # # Prepare inputs and background dataset
    # discriminator_input_array_flat = ...  # Flattened input data for the discriminator
    # discriminator_background = shap.kmeans(discriminator_input_array_flat, 50)

    # # Create SHAP explainer
    # discriminator_explainer = shap.KernelExplainer(discriminator_predict, discriminator_background)

    # # Select test samples
    # discriminator_test_samples = discriminator_input_array_flat[:10]

    # # Compute SHAP values
    # discriminator_shap_values = discriminator_explainer.shap_values(discriminator_test_samples)

    # # Visualize SHAP values
    # shap.summary_plot(discriminator_shap_values, discriminator_test_samples, feature_names=discriminator_feature_names)


if __name__ == "__main__":
    interpret_model()
