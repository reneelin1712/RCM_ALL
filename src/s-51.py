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
    state_list = []
    action_list = []
    destination_list = []
    input_data_disc_list = []
    next_state_list = []

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
            action_list.append(action)
            destination_list.append(destination_node)
            next_state_list.append(next_state)

            # Prepare input data for discriminator network
            input_data_disc = get_discriminator_input(discriminator_net, state, destination_node, device)
            input_data_disc_np = input_data_disc.cpu().detach().numpy().squeeze(0)
            input_data_disc_list.append(input_data_disc_np)

    # Convert input_data_list to numpy array
    input_array = np.array(input_data_list)  # Shape: [num_samples, channels, height, width]
    input_array_disc = np.array(input_data_disc_list)

    # Prepare input tensors for SHAP (Policy Network)
    num_samples_total = input_array.shape[0]
    num_channels = input_array.shape[1]
    height = input_array.shape[2]
    width = input_array.shape[3]

    # Reshape input data to [num_samples_total, num_features]
    input_array_flat = input_array.reshape(num_samples_total, -1)
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
        outputs = np.zeros(num_samples)
        for i in range(num_samples):
            # Reconstruct input_data tensor
            input_data_np = input_array_flat[i].reshape(num_channels, height, width)
            input_data_tensor = torch.tensor(input_data_np, dtype=torch.float32).unsqueeze(0).to(device)
            state = state_list[i]
            action_taken = action_list[i]
            state_tensor = torch.tensor([state], dtype=torch.long).to(device)
            with torch.no_grad():
                output = policy_net.forward(input_data_tensor)
                x_mask = policy_net.policy_mask[state_tensor]
                output = output.masked_fill((1 - x_mask).bool(), -1e32)
                action_probs = torch.softmax(output, dim=1)
                outputs[i] = action_probs.cpu().numpy().flatten()[action_taken]
        return outputs  # Shape: [num_samples]

    # Create background dataset
    policy_background = shap.kmeans(input_array_flat, 50)

    # Select test samples
    policy_test_samples = input_array_flat[0:1000]  # Adjust as needed

    # Update num_samples to match the number of test samples
    num_samples = policy_test_samples.shape[0]
    # Adjust state_list and action_list accordingly
    state_list_test = state_list[:num_samples]
    action_list_test = action_list[:num_samples]

    # Create the SHAP explainer for policy network
    policy_explainer = shap.KernelExplainer(policy_predict, policy_background)

    # Compute SHAP values for policy network
    policy_shap_values = policy_explainer.shap_values(policy_test_samples)

    # Reshape SHAP values back to [num_samples, num_channels, height, width]
    policy_shap_values_reshaped = policy_shap_values.reshape(num_samples, num_channels, height, width)

    # Aggregate SHAP values over spatial dimensions
    policy_shap_values_per_feature = np.sum(policy_shap_values_reshaped, axis=(2, 3))  # Shape: [num_samples, num_channels]

    # Aggregate test samples over spatial dimensions for visualization
    policy_test_samples_reshaped = policy_test_samples.reshape(num_samples, num_channels, height, width)
    policy_test_samples_per_feature = np.mean(policy_test_samples_reshaped, axis=(2, 3))  # Shape: [num_samples, num_channels]

    # Use the original feature names
    feature_names_channels = feature_names  # List of length num_channels

    # Visualize SHAP values for policy network
    shap.summary_plot(policy_shap_values_per_feature, policy_test_samples_per_feature, feature_names=feature_names_channels, show=False)
    # Save the SHAP summary plot
    plt.savefig('shap_img/discriminator_shap_summary_plot.png', bbox_inches='tight')
    plt.close()

    # Optionally, aggregate SHAP values over samples for global importance
    global_shap_values = np.mean(policy_shap_values_per_feature, axis=0)  # Shape: [num_channels]

    # Create a DataFrame for visualization
    df_global_shap = pd.DataFrame({
        'Feature': feature_names_channels,
        'Mean SHAP Value': global_shap_values
    })

    # Sort by absolute SHAP Value
    # df_global_shap['abs_SHAP_Value'] = np.abs(df_global_shap['Mean SHAP Value'])
    # df_global_shap_sorted = df_global_shap.sort_values('abs_SHAP_Value', ascending=False)
    df_global_shap_sorted = df_global_shap.sort_values('Mean SHAP Value', ascending=False)


    # Plot the global feature importance
    plt.figure(figsize=(10, 6))
    plt.barh(df_global_shap_sorted['Feature'], df_global_shap_sorted['Mean SHAP Value'], color='skyblue')
    plt.xlabel('Mean SHAP Value')
    plt.title('Policy Network Global Feature Importance')
    plt.gca().invert_yaxis()  # Highest importance at the top
    plt.tight_layout()

    # Save the plot
    plt.savefig('shap_img/policy_global_feature_importance.png')
    # plt.show()

    # Optionally, save the DataFrame to CSV
    df_global_shap_sorted.to_csv('shap_img/policy_global_feature_importance.csv', index=False)

    ### Now for the Discriminator Network ###

    # Prepare inputs for discriminator network
    input_array_disc = np.array(input_data_disc_list)  # Shape: [num_samples_total, channels, height, width]
    num_samples_total_disc = input_array_disc.shape[0]
    num_channels_disc = input_array_disc.shape[1]
    height_disc = input_array_disc.shape[2]
    width_disc = input_array_disc.shape[3]

    # Flatten input data for SHAP
    input_array_disc_flat = input_array_disc.reshape(num_samples_total_disc, -1)
    num_features_disc = input_array_disc_flat.shape[1]

    # Expand feature names to match flattened input for discriminator
    expanded_feature_names_disc = []
    for c in range(num_channels_disc):
        for h in range(height_disc):
            for w in range(width_disc):
                expanded_feature_names_disc.append(f"{feature_names[c]}_{h}_{w}")

    # Define the prediction function for the discriminator network
    def discriminator_predict(input_array_disc_flat):
        # Input array shape: [num_samples, num_features]
        num_samples = input_array_disc_flat.shape[0]
        outputs = np.zeros(num_samples)
        for i in range(num_samples):
            # Reconstruct input_data_disc tensor
            input_data_disc_np = input_array_disc_flat[i].reshape(num_channels_disc, height_disc, width_disc)
            input_data_disc_tensor = torch.tensor(input_data_disc_np, dtype=torch.float32).unsqueeze(0).to(device)

            # Get action
            act_taken = action_list[i]
            act_tensor = torch.tensor([act_taken], dtype=torch.long).to(device)
            act_one_hot = F.one_hot(act_tensor, num_classes=discriminator_net.action_num).float()

            # Get state, des, next_state
            state_tensor = torch.tensor([state_list[i]], dtype=torch.long).to(device)
            des_tensor = torch.tensor([destination_list[i]], dtype=torch.long).to(device)
            next_state_tensor = torch.tensor([next_state_list[i]], dtype=torch.long).to(device)

            # Pass through the discriminator network
            with torch.no_grad():
                x = input_data_disc_tensor
                x = discriminator_net.pool(F.leaky_relu(discriminator_net.conv1(x), 0.2))
                x = F.leaky_relu(discriminator_net.conv2(x), 0.2)
                x = x.view(-1, 30)  # x shape: [batch_size, 30]

                # Concatenate x and act_one_hot
                x = torch.cat([x, act_one_hot], 1)  # x shape: [batch_size, 30 + num_classes]

                x = F.leaky_relu(discriminator_net.fc1(x), 0.2)
                x = F.leaky_relu(discriminator_net.fc2(x), 0.2)
                rs = discriminator_net.fc3(x)

                # Compute hs and hs_next
                x_state = discriminator_net.process_state_features(state_tensor, des_tensor)
                x_state = F.leaky_relu(discriminator_net.h_fc1(x_state), 0.2)
                x_state = F.leaky_relu(discriminator_net.h_fc2(x_state), 0.2)
                x_state = discriminator_net.h_fc3(x_state)

                next_x_state = discriminator_net.process_state_features(next_state_tensor, des_tensor)
                next_x_state = F.leaky_relu(discriminator_net.h_fc1(next_x_state), 0.2)
                next_x_state = F.leaky_relu(discriminator_net.h_fc2(next_x_state), 0.2)
                next_x_state = discriminator_net.h_fc3(next_x_state)

                f = rs + discriminator_net.gamma * next_x_state - x_state
                f = f.squeeze(-1)  # Shape: [batch_size]

                outputs[i] = f.cpu().numpy().flatten()[0]
        return outputs  # Shape: [num_samples]

    # Create background dataset for discriminator network
    discriminator_background = shap.kmeans(input_array_disc_flat, 50)

    # Select test samples for discriminator network
    discriminator_test_samples = input_array_disc_flat[0:1000]  # Adjust as needed

    # Update num_samples to match the number of test samples
    num_samples_disc = discriminator_test_samples.shape[0]
    # Adjust action_list and other lists accordingly
    action_list_disc = action_list[:num_samples_disc]
    state_list_disc = state_list[:num_samples_disc]
    destination_list_disc = destination_list[:num_samples_disc]
    next_state_list_disc = next_state_list[:num_samples_disc]

    # Create the SHAP explainer for discriminator network
    discriminator_explainer = shap.KernelExplainer(discriminator_predict, discriminator_background)

    # Compute SHAP values for discriminator network
    discriminator_shap_values = discriminator_explainer.shap_values(discriminator_test_samples)

    # Reshape SHAP values back to [num_samples_disc, num_channels_disc, height_disc, width_disc]
    discriminator_shap_values_reshaped = discriminator_shap_values.reshape(num_samples_disc, num_channels_disc, height_disc, width_disc)

    # Aggregate SHAP values over spatial dimensions
    discriminator_shap_values_per_feature = np.sum(discriminator_shap_values_reshaped, axis=(2, 3))  # Shape: [num_samples_disc, num_channels_disc]

    # Aggregate test samples over spatial dimensions for visualization
    discriminator_test_samples_reshaped = discriminator_test_samples.reshape(num_samples_disc, num_channels_disc, height_disc, width_disc)
    discriminator_test_samples_per_feature = np.mean(discriminator_test_samples_reshaped, axis=(2, 3))  # Shape: [num_samples_disc, num_channels_disc]

    # Visualize SHAP values for discriminator network
    shap.summary_plot(discriminator_shap_values_per_feature, discriminator_test_samples_per_feature, feature_names=feature_names_channels,show=False)
    # Save the SHAP summary plot for the policy network
    plt.savefig('shap_img/policy_shap_summary_plot.png', bbox_inches='tight')
    plt.close()

    # Optionally, aggregate SHAP values over samples for global importance
    global_shap_values_disc = np.mean(discriminator_shap_values_per_feature, axis=0)  # Shape: [num_channels_disc]

    # Create a DataFrame for visualization
    df_global_shap_disc = pd.DataFrame({
        'Feature': feature_names_channels,
        'Mean SHAP Value': global_shap_values_disc
    })

    # Sort by absolute SHAP Value
    # df_global_shap_disc['abs_SHAP_Value'] = np.abs(df_global_shap_disc['Mean SHAP Value'])
    # df_global_shap_disc_sorted = df_global_shap_disc.sort_values('abs_SHAP_Value', ascending=False)
    df_global_shap_disc_sorted = df_global_shap_disc.sort_values('Mean SHAP Value', ascending=False)

    # Plot the global feature importance for discriminator network
    plt.figure(figsize=(10, 6))
    plt.barh(df_global_shap_disc_sorted['Feature'], df_global_shap_disc_sorted['Mean SHAP Value'], color='skyblue')
    plt.xlabel('Mean SHAP Value')
    plt.title('Discriminator Network Global Feature Importance')
    plt.gca().invert_yaxis()  # Highest importance at the top
    plt.tight_layout()

    # Save the plot
    plt.savefig('shap_img/discriminator_global_feature_importance.png')
    # plt.show()

    # Optionally, save the DataFrame to CSV
    df_global_shap_disc_sorted.to_csv('shap_img/discriminator_global_feature_importance.csv', index=False)

if __name__ == "__main__":
    interpret_model()
