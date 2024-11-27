import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import pandas as pd
import os

# Captum imports for interpretability
# from captum.attr import IntegratedGradients
from captum.attr import GradientShap

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

    # Extend feature names to include action features for the discriminator
    action_feature_names = [f'Action_{i}' for i in range(discriminator_net.action_num)]
    combined_feature_names = feature_names + action_feature_names

    # Read generated trajectories
    df_generated = pd.read_csv(generated_trajs_csv)[0:30]

    # Ensure output directories exist
    output_dir = f'output_img_{model_name}_20241127_final'
    output_csv_dir = os.path.join(output_dir, 'csv')
    output_png_dir = os.path.join(output_dir, 'png')
    os.makedirs(output_csv_dir, exist_ok=True)
    os.makedirs(output_png_dir, exist_ok=True)

    # Initialize dictionaries to store aggregated feature importance separately
    aggregated_importance_policy = {
        'highest_reward': [],
        'lowest_reward': [],
        'divergence': [],
        'convergence': []
    }

    aggregated_importance_discriminator = {
        'highest_reward': [],
        'lowest_reward': [],
        'divergence': [],
        'convergence': []
    }

    # Initialize a list to store overlap counts
    overlap_counts = {
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

        # Initialize lists to store rewards for plotting
        trajectory_rewards = rewards_list

        # Plot rewards over trajectory
        plt.figure(figsize=(8, 6))
        plt.plot(range(1, len(trajectory_rewards) + 1), trajectory_rewards, marker='o', label='Reward')

        # Annotate divergence and convergence steps
        for key, step_idx in indices_to_analyze.items():
            if key in ['divergence', 'convergence']:
                color = 'red' if key == 'divergence' else 'green'
                plt.axvline(x=step_idx + 1, color=color, linestyle='--', label=f'{key.capitalize()} Step')
                y_position = max(trajectory_rewards) * 0.95
                plt.text(step_idx + 1, y_position, f'{key.capitalize()}',
                         rotation=90, verticalalignment='top', color=color)

        plt.title(f'Rewards over Trajectory Steps (Trajectory {idx+1})')
        plt.xlabel('Step')
        plt.ylabel('Reward')
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(output_png_dir, f'trajectory_{idx+1}_rewards_over_trajectory.png'))
        plt.close()

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

            # Compute attributions for policy network
            def forward_func_policy(input_data):
                x = input_data
                x = policy_net.forward(x)
                x_mask = policy_net.policy_mask[state_tensor]
                x = x.masked_fill((1 - x_mask).bool(), -1e32)
                action_probs = torch.softmax(x, dim=1)
                return action_probs  # Returns tensor of shape [batch_size, num_actions]

            # Integrated Gradients for policy network
            ig_policy = IntegratedGradients(forward_func_policy)
            attributions_ig_policy = ig_policy.attribute(input_data, target=action_taken)
            attributions_ig_policy_np = attributions_ig_policy.squeeze().cpu().detach().numpy()

            # Aggregate attributions for policy network (Preserving sign)
            channel_importance_policy = np.sum(attributions_ig_policy_np, axis=(1, 2))

            # Append to aggregated importance
            aggregated_importance_policy[key].append(channel_importance_policy)

            # Create DataFrame for policy network
            importance_scores_policy = channel_importance_policy
            ranked_indices_policy = np.argsort(-np.abs(importance_scores_policy))  # Sort by absolute value
            sorted_features_policy = [feature_names[i] for i in ranked_indices_policy]
            sorted_importance_policy = importance_scores_policy[ranked_indices_policy]

            feature_importance_policy_df = pd.DataFrame({
                'Feature': sorted_features_policy,
                'Importance Score': sorted_importance_policy
            })

            # Save DataFrame to CSV
            feature_importance_policy_df.to_csv(
                os.path.join(output_csv_dir, f'trajectory_{idx+1}_{key}_policy_ig_feature_importance.csv'), index=False
            )

            # Now compute attributions for the discriminator network
            # Prepare action tensor
            act_tensor = torch.tensor([action_taken], dtype=torch.long).to(device)
            act_one_hot = F.one_hot(act_tensor, num_classes=discriminator_net.action_num).float()
            act_one_hot.requires_grad = True

            # Process features for discriminator
            input_data_disc = discriminator_net.process_neigh_features(state_tensor, des_tensor)
            input_data_disc.requires_grad = True

            next_state_tensor = torch.tensor([next_state_taken], dtype=torch.long).to(device)

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
                f = f.squeeze(-1)  # Ensure f is of shape [batch_size]
                return f

            # Integrated Gradients for discriminator network
            ig_disc = IntegratedGradients(forward_func_discriminator)
            attributions_ig_disc = ig_disc.attribute((input_data_disc, act_one_hot))
            attributions_input = attributions_ig_disc[0].squeeze().cpu().detach().numpy()
            attributions_action = attributions_ig_disc[1].squeeze().cpu().detach().numpy()

            # Aggregate attributions
            channel_importance_input = np.sum(attributions_input, axis=(1, 2))
            channel_importance_action = attributions_action  # Already a vector

            # Combine input feature attributions and action attributions
            combined_importance = np.concatenate([channel_importance_input, channel_importance_action])

            # Append to aggregated importance
            aggregated_importance_discriminator[key].append(combined_importance)

            # Create DataFrame for discriminator network
            importance_scores_disc = combined_importance
            ranked_indices_disc = np.argsort(-np.abs(importance_scores_disc))
            sorted_features_disc = [combined_feature_names[i] for i in ranked_indices_disc]
            sorted_importance_disc = importance_scores_disc[ranked_indices_disc]

            feature_importance_disc_df = pd.DataFrame({
                'Feature': sorted_features_disc,
                'Importance Score': sorted_importance_disc
            })

            # Save DataFrame to CSV
            feature_importance_disc_df.to_csv(
                os.path.join(output_csv_dir, f'trajectory_{idx+1}_{key}_discriminator_ig_feature_importance.csv'),
                index=False
            )

            # Compare top N features
            N = 3  # Top N features to compare
            top_features_policy = set(sorted_features_policy[:N])
            top_features_discriminator = set(sorted_features_disc[:N])
            overlap = top_features_policy.intersection(top_features_discriminator)
            overlap_count = len(overlap)

            # Store overlap count
            overlap_counts[key].append(overlap_count)

            # Save overlap information
            # Create a DataFrame that maps features to their presence in top N and overlap
            all_top_features = list(top_features_policy.union(top_features_discriminator))
            data = []
            for feature in all_top_features:
                data.append({
                    'Feature': feature,
                    'In Policy Top N': feature in top_features_policy,
                    'In Discriminator Top N': feature in top_features_discriminator,
                    'In Overlap': feature in overlap
                })
            overlap_df = pd.DataFrame(data)

            overlap_df.to_csv(
                os.path.join(output_csv_dir, f'trajectory_{idx+1}_{key}_feature_overlap.csv'), index=False
            )

            # Create combined plot of feature importance (including sign)
            fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(20, 6))

            # Plot Policy Network Feature Importance
            axes[0].barh(sorted_features_policy, sorted_importance_policy, color=['green' if v >= 0 else 'red' for v in sorted_importance_policy])
            axes[0].set_xlabel('Importance Score')
            axes[0].set_title('Policy Network IG Feature Importance')
            axes[0].invert_yaxis()  # Highest importance at the top

            # Plot Discriminator Network Feature Importance
            axes[1].barh(sorted_features_disc, sorted_importance_disc, color=['green' if v >= 0 else 'red' for v in sorted_importance_disc])
            axes[1].set_xlabel('Importance Score')
            axes[1].set_title('Discriminator IG Feature Importance')
            axes[1].invert_yaxis()  # Highest importance at the top

            # Set the overall title
            fig.suptitle(f'Trajectory {idx+1} - Step {step_idx+1} ({key.capitalize()})', fontsize=16)

            plt.tight_layout(rect=[0, 0.03, 1, 0.95])  # Adjust layout to make room for suptitle

            # Save the combined figure
            plt.savefig(os.path.join(output_png_dir, f'trajectory_{idx+1}_{key}_combined_feature_importance.png'))
            plt.close()

    # After processing all trajectories, compute average feature importance
    for key in aggregated_importance_policy.keys():
        if aggregated_importance_policy[key]:
            # Stack all importance scores and compute mean
            importance_array_policy = np.stack(aggregated_importance_policy[key], axis=0)
            mean_importance_policy = np.mean(importance_array_policy, axis=0)

            # Policy Network Mean Importance
            importance_scores_policy = mean_importance_policy
            ranked_indices_policy = np.argsort(-np.abs(importance_scores_policy))  # Sort by absolute value
            sorted_features_policy = [feature_names[i] for i in ranked_indices_policy]
            sorted_importance_policy = importance_scores_policy[ranked_indices_policy]

            # Create DataFrame for policy network
            feature_importance_policy_df = pd.DataFrame({
                'Feature': sorted_features_policy,
                'Mean Importance Score': sorted_importance_policy
            })

            # Save DataFrame to CSV
            feature_importance_policy_df.to_csv(
                os.path.join(output_csv_dir, f'aggregated_{key}_policy_ig_feature_importance.csv'), index=False
            )

            # Now process the discriminator aggregated importance
            importance_array_disc = np.stack(aggregated_importance_discriminator[key], axis=0)
            mean_importance_disc = np.mean(importance_array_disc, axis=0)

            # Discriminator Network Mean Importance
            importance_scores_disc = mean_importance_disc
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

            # Compare top N features in aggregated importance
            N = 3  # Top N features to compare
            top_features_policy_agg = set(sorted_features_policy[:N])
            top_features_discriminator_agg = set(sorted_features_disc[:N])
            overlap_agg = top_features_policy_agg.intersection(top_features_discriminator_agg)
            overlap_count_agg = len(overlap_agg)

            # Save aggregated overlap information
            all_top_features_agg = list(top_features_policy_agg.union(top_features_discriminator_agg))
            data_agg = []
            for feature in all_top_features_agg:
                data_agg.append({
                    'Feature': feature,
                    'In Policy Top N': feature in top_features_policy_agg,
                    'In Discriminator Top N': feature in top_features_discriminator_agg,
                    'In Overlap': feature in overlap_agg
                })
            overlap_agg_df = pd.DataFrame(data_agg)

            overlap_agg_df.to_csv(
                os.path.join(output_csv_dir, f'aggregated_{key}_feature_overlap.csv'), index=False
            )

            # Create combined plot of aggregated feature importance (including sign)
            fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(20, 6))

            # Plot Policy Network Mean Feature Importance
            axes[0].barh(sorted_features_policy, sorted_importance_policy, color=['green' if v >= 0 else 'red' for v in sorted_importance_policy])
            axes[0].set_xlabel('Mean Importance Score')
            axes[0].set_title('Aggregated Policy Network IG Feature Importance')
            axes[0].invert_yaxis()  # Highest importance at the top

            # Plot Discriminator Network Mean Feature Importance
            axes[1].barh(sorted_features_disc, sorted_importance_disc, color=['green' if v >= 0 else 'red' for v in sorted_importance_disc])
            axes[1].set_xlabel('Mean Importance Score')
            axes[1].set_title('Aggregated Discriminator IG Feature Importance')
            axes[1].invert_yaxis()  # Highest importance at the top

            # Set the overall title
            fig.suptitle(f'Aggregated Feature Importance - {key.capitalize()} Steps', fontsize=16)

            plt.tight_layout(rect=[0, 0.03, 1, 0.95])  # Adjust layout to make room for suptitle

            # Save the combined figure
            plt.savefig(os.path.join(output_png_dir, f'aggregated_{key}_combined_feature_importance.png'))
            plt.close()

    # Output overlap statistics
    for key in overlap_counts.keys():
        if overlap_counts[key]:
            avg_overlap = np.mean(overlap_counts[key])
            print(f"Average top-{N} feature overlap at {key} steps: {avg_overlap:.2f}")

    print("Interpretation complete. Results saved in the output directories.")

if __name__ == "__main__":
    interpret_model()
