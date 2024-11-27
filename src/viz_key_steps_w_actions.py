import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import pandas as pd
import os

# Captum imports for interpretability
from captum.attr import IntegratedGradients

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

    # Extend feature names to include action features
    action_feature_names = [f'Action_{i}' for i in range(discriminator_net.action_num)]
    combined_feature_names = feature_names + action_feature_names

    # Read generated trajectories
    df_generated = pd.read_csv(generated_trajs_csv)[0:30]

    # Ensure output directories exist
    output_dir = f'output_img_{model_name}'
    output_csv_dir = os.path.join(output_dir, 'csv')
    output_png_dir = os.path.join(output_dir, 'png')
    os.makedirs(output_csv_dir, exist_ok=True)
    os.makedirs(output_png_dir, exist_ok=True)

    # Initialize dictionaries to store aggregated feature importance separately
    aggregated_importance_discriminator = {
        'highest_reward': [],
        'lowest_reward': [],
        'divergence': [],
        'convergence': []
    }

    # Loop over each trajectory in the DataFrame
    for idx, row in df_generated.iterrows():
        origin = row['origin']
        destination = row['destination']
        trajectory_str = row['generated_trajectory']
        actions_taken = eval(row['actions_taken'])  # Convert string representation of list to actual list
        rewards_list = eval(row['rewards_per_step'])  # Same for rewards
        highest_reward_index = row['highest_reward_index']
        lowest_reward_index = row['lowest_reward_index']
        divergence_index = row['divergence_index']
        convergence_index = row['convergence_index']

        # Convert trajectory string to list of integers
        trajectory = [int(s) for s in trajectory_str.strip().split('_')]
        states_list = trajectory[:-1]
        destination_node = trajectory[-1]
        total_steps = len(states_list)

        # Create a list of indices to analyze
        indices_to_analyze = {
            'highest_reward': highest_reward_index,
            'lowest_reward': lowest_reward_index,
            'divergence': divergence_index,
            'convergence': convergence_index
        }

        # Remove invalid indices (e.g., NaN)
        indices_to_analyze = {k: int(v)-1 for k, v in indices_to_analyze.items() if not pd.isnull(v)}
        if not indices_to_analyze:
            continue  # Skip if there are no valid indices

        # Loop over each index to analyze
        for key, step_idx in indices_to_analyze.items():
            if step_idx < 0 or step_idx >= total_steps:
                continue  # Skip invalid indices

            state = states_list[step_idx]
            action_taken = actions_taken[step_idx]
            next_state_taken = trajectory[step_idx + 1]

            # Prepare input data for policy network
            input_data = get_cnn_input(policy_net, state, destination_node, device)
            input_data.requires_grad = True

            state_tensor = torch.tensor([state], dtype=torch.long).to(device)
            des_tensor = torch.tensor([destination_node], dtype=torch.long).to(device)

            # Get valid actions at the current state
            valid_actions_mask = policy_net.policy_mask[state_tensor].squeeze()
            valid_actions_indices = torch.nonzero(valid_actions_mask, as_tuple=False).squeeze()

            # Ensure valid_actions_indices is iterable
            if valid_actions_indices.dim() == 0:
                valid_actions_indices = valid_actions_indices.unsqueeze(0)

            # Initialize a list to store attributions for each action
            action_attributions_list = []

            # Iterate over all valid actions
            for action_idx in valid_actions_indices:
                action_idx = action_idx.item()

                # Get the next state for this action
                next_state = env.get_next_state(state, action_idx)
                next_state_tensor = torch.tensor([next_state], dtype=torch.long).to(device)

                # Prepare action tensor
                act_tensor = torch.tensor([action_idx], dtype=torch.long).to(device)
                act_one_hot = F.one_hot(act_tensor, num_classes=discriminator_net.action_num).float()
                act_one_hot.requires_grad = True

                # Process features for discriminator
                input_data_disc = discriminator_net.process_neigh_features(state_tensor, des_tensor)
                input_data_disc.requires_grad = True

                def forward_func_discriminator(input_data, x_act):
                    # Discriminator computations
                    x = input_data
                    x = discriminator_net.pool(F.leaky_relu(discriminator_net.conv1(x), 0.2))
                    x = F.leaky_relu(discriminator_net.conv2(x), 0.2)
                    x = x.view(-1, 30)  # x shape: [batch_size, 30]

                    # x_act is act_one_hot
                    if x_act.dim() == 1:
                        x_act = x_act.unsqueeze(0)  # x_act shape: [1, num_classes]

                    # Expand x_act to match the batch size of x
                    batch_size = x.shape[0]
                    x_act = x_act.expand(batch_size, -1)  # x_act shape: [batch_size, num_classes]

                    # Concatenate x and x_act
                    x = torch.cat([x, x_act], 1)  # x shape: [batch_size, 30 + num_classes]

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
                    return f

                # Integrated Gradients for discriminator network
                ig_disc = IntegratedGradients(forward_func_discriminator)
                attributions_ig_disc = ig_disc.attribute((input_data_disc, act_one_hot))

                # Split attributions
                attributions_input = attributions_ig_disc[0].squeeze().cpu().detach().numpy()
                attributions_action = attributions_ig_disc[1].squeeze().cpu().detach().numpy()

                # Aggregate attributions
                channel_importance_input = np.sum(attributions_input, axis=(1, 2))
                channel_importance_action = attributions_action  # Already a vector

                # Combine input feature attributions and action attributions
                combined_importance = np.concatenate([channel_importance_input, channel_importance_action])

                # Store attributions with the action index
                action_attributions_list.append({
                    'action_index': action_idx,
                    'attributions': combined_importance
                })

            # Now, you can analyze the attributions for all actions
            # For example, create a DataFrame to compare them

            # Aggregate attributions across all actions at this step
            all_attributions = np.stack([attr['attributions'] for attr in action_attributions_list])
            mean_importance_disc = np.mean(all_attributions, axis=0)

            # Store mean importance for this step and key
            aggregated_importance_discriminator[key].append(mean_importance_disc)

            # You can also save the aggregated attributions
            importance_scores_disc = mean_importance_disc
            ranked_indices_disc = np.argsort(-np.abs(importance_scores_disc))
            sorted_features_disc = [combined_feature_names[i] for i in ranked_indices_disc]
            sorted_importance_disc = importance_scores_disc[ranked_indices_disc]

            # Create DataFrame for aggregated attributions
            feature_importance_disc_df = pd.DataFrame({
                'Feature': sorted_features_disc,
                'Mean Importance Score': sorted_importance_disc
            })

            # Save DataFrame to CSV
            feature_importance_disc_df.to_csv(
                os.path.join(output_csv_dir, f'trajectory_{idx+1}_{key}_aggregated_discriminator_ig_feature_importance.csv'),
                index=False
            )

            # Optionally, plot the aggregated feature importances
            plt.figure(figsize=(10, 6))
            plt.barh(sorted_features_disc, sorted_importance_disc, color=['green' if v >= 0 else 'red' for v in sorted_importance_disc])
            plt.xlabel('Mean Importance Score')
            plt.title(f'Aggregated Discriminator IG Feature Importance - {key.capitalize()} (Trajectory {idx+1})')
            plt.gca().invert_yaxis()  # Highest importance at the top
            plt.tight_layout()
            plt.savefig(os.path.join(output_png_dir, f'trajectory_{idx+1}_{key}_aggregated_feature_importance.png'))
            plt.close()

    # After processing all trajectories, compute average feature importance
    for key in aggregated_importance_discriminator.keys():
        if aggregated_importance_discriminator[key]:
            # Stack all mean importances and compute overall mean
            importance_array_disc = np.stack(aggregated_importance_discriminator[key], axis=0)
            overall_mean_importance_disc = np.mean(importance_array_disc, axis=0)

            # Discriminator Network Mean Importance
            importance_scores_disc = overall_mean_importance_disc
            ranked_indices_disc = np.argsort(-np.abs(importance_scores_disc))
            sorted_features_disc = [combined_feature_names[i] for i in ranked_indices_disc]
            sorted_importance_disc = importance_scores_disc[ranked_indices_disc]

            # Create DataFrame
            feature_importance_disc_df = pd.DataFrame({
                'Feature': sorted_features_disc,
                'Mean Importance Score': sorted_importance_disc
            })

            # Save DataFrame to CSV
            feature_importance_disc_df.to_csv(
                os.path.join(output_csv_dir, f'aggregated_{key}_discriminator_ig_feature_importance.csv'),
                index=False
            )

            # Plot the aggregated feature importances
            plt.figure(figsize=(10, 6))
            plt.barh(sorted_features_disc, sorted_importance_disc, color=['green' if v >= 0 else 'red' for v in sorted_importance_disc])
            plt.xlabel('Mean Importance Score')
            plt.title(f'Overall Aggregated Discriminator IG Feature Importance - {key.capitalize()}')
            plt.gca().invert_yaxis()  # Highest importance at the top
            plt.tight_layout()
            plt.savefig(os.path.join(output_png_dir, f'aggregated_{key}_overall_feature_importance.png'))
            plt.close()

    print("Interpretation complete. Results saved in the output directories.")

if __name__ == "__main__":
    interpret_model()

