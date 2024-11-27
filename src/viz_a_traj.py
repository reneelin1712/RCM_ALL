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
    load_test_traj,
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


    # Path settings (adjust paths as necessary)
    model_path = "../trained_models/base/bleu90.pt"
    edge_p = "../data/base/edge.txt"
    network_p = "../data/base/transit.npy"
    path_feature_p = "../data/base/feature_od.npy"
    test_p = "../data/base/cross_validation/test_CV0.csv"

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
    policy_net, value_net, discriminator_net = load_model(
        model_path, device, env, path_feature_pad, edge_feature_pad
    )

    # Extend feature names to include action features for the discriminator
    action_feature_names = [f'Action_{i}' for i in range(discriminator_net.action_num)]
    combined_feature_names = feature_names + action_feature_names


    # Manually set the trajectory
    manual_traj_str = "1_338_154_174_364_178_180_337_182_186_347_403_243_212_25_252_245_37_54_358_340_33_343_377_39_314_49_313_301_40"
    manual_traj = manual_traj_str.split('_')  # This will create a list of node IDs as strings
    test_trajs = [manual_traj]

    # Prepare input data
    states_list = [int(s) for s in manual_traj[:-1]]  # All states except the last one
    destination = int(manual_traj[-1])  # The destination is the last state

    # Ensure output directories exist
    output_dir = 'output_img_step_by_step'
    os.makedirs(output_dir, exist_ok=True)

    # Initialize lists to store attributions and importance scores
    attributions_ig_list = []
    channel_importance_ig_list = []
    disc_attributions_ig_list = []
    channel_importance_ig_disc_list = []
    rewards_list = []
    actions_list = []

    # Loop over the states in the trajectory
    for idx, state in enumerate(states_list):
        # Get CNN input for policy network
        input_data = get_cnn_input(policy_net, state, destination, device)
        input_data.requires_grad = True

        state_tensor = torch.tensor([state], dtype=torch.long).to(device)
        des_tensor = torch.tensor([destination], dtype=torch.long).to(device)

        # Compute attributions for policy network
        def forward_func_policy(input_data):
            x = input_data
            x = policy_net.forward(x)
            x_mask = policy_net.policy_mask[state_tensor]
            x = x.masked_fill((1 - x_mask).bool(), -1e32)
            action_probs = torch.softmax(x, dim=1)
            return action_probs

        with torch.no_grad():
            output = policy_net.forward(input_data)
            x_mask = policy_net.policy_mask[state_tensor]
            output = output.masked_fill((1 - x_mask).bool(), -1e32)
            action_probs = torch.softmax(output, dim=1)
            action_taken = torch.argmax(action_probs, dim=1).item()
            log_pi = torch.log(action_probs[0, action_taken])

        actions_list.append(action_taken)
        next_state = env.state_action[state][action_taken]
        next_state_tensor = torch.tensor([next_state], dtype=torch.long).to(device)

        ig_policy = IntegratedGradients(forward_func_policy)
        attributions_ig_policy = ig_policy.attribute(input_data, target=action_taken)
        attributions_ig_policy_np = attributions_ig_policy.squeeze().cpu().detach().numpy()
        attributions_ig_list.append(attributions_ig_policy_np)

        # Aggregate attributions for policy network (Preserving sign)
        channel_importance_policy = np.sum(attributions_ig_policy_np, axis=(1, 2))
        channel_importance_ig_list.append(channel_importance_policy)

        # Now compute attributions for the discriminator network
        # Prepare action tensor
        act_tensor = torch.tensor([action_taken], dtype=torch.long).to(device)
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
            f = f.squeeze(-1)  # Ensure f is of shape [batch_size]
            return f

        ig_disc = IntegratedGradients(forward_func_discriminator)
        attributions_ig_disc = ig_disc.attribute((input_data_disc, act_one_hot))
        attributions_input = attributions_ig_disc[0].squeeze().cpu().detach().numpy()
        attributions_action = attributions_ig_disc[1].squeeze().cpu().detach().numpy()

        # Aggregate attributions
        channel_importance_input = np.sum(attributions_input, axis=(1, 2))
        channel_importance_action = attributions_action  # Already a vector

        # Combine input feature attributions and action attributions
        combined_importance = np.concatenate([channel_importance_input, channel_importance_action])
        channel_importance_ig_disc_list.append(combined_importance)

        # Calculate reward
        with torch.no_grad():
            reward = discriminator_net.calculate_reward(
                state_tensor, des_tensor, act_tensor, log_pi, next_state_tensor
            )
            rewards_list.append(reward.item())

    # After the loop, process and visualize the attributions and feature importance rankings
    num_steps = len(channel_importance_ig_list)
    for idx in range(num_steps):
        action = actions_list[idx]

        # Policy Network - Integrated Gradients
        importance_scores = channel_importance_ig_list[idx]
        ranked_indices = np.argsort(-np.abs(importance_scores))
        sorted_features = [feature_names[i] for i in ranked_indices]
        sorted_importance = importance_scores[ranked_indices]

        # Create DataFrame
        feature_importance_df = pd.DataFrame({
            'Feature': sorted_features,
            'Importance Score': sorted_importance
        })

        # Save DataFrame to CSV
        feature_importance_df.to_csv(
            os.path.join(output_dir, f'policy_ig_feature_importance_step_{idx+1}.csv'), index=False
        )

        # Plot Feature Importance (Include Sign)
        plt.figure(figsize=(10, 6))
        colors = ['green' if val >= 0 else 'red' for val in sorted_importance]
        plt.barh(sorted_features[::-1], sorted_importance[::-1], color=colors[::-1])
        plt.xlabel('Importance Score')
        plt.title(f'Policy Network IG Feature Importance - Step {idx+1} - Action {action}')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f'policy_ig_feature_importance_step_{idx+1}.png'))
        plt.close()

        # Discriminator Network - Integrated Gradients
        importance_scores_disc = channel_importance_ig_disc_list[idx]
        ranked_indices_disc = np.argsort(-np.abs(importance_scores_disc))
        sorted_features_disc = [combined_feature_names[i] for i in ranked_indices_disc]
        sorted_importance_disc = importance_scores_disc[ranked_indices_disc]

        # Create DataFrame
        feature_importance_disc_df = pd.DataFrame({
            'Feature': sorted_features_disc,
            'Importance Score': sorted_importance_disc
        })

        # Save DataFrame to CSV
        feature_importance_disc_df.to_csv(
            os.path.join(output_dir, f'discriminator_ig_feature_importance_step_{idx+1}.csv'), index=False
        )

        # Plot Feature Importance (Include Sign)
        plt.figure(figsize=(10, 6))
        colors_disc = ['green' if val >= 0 else 'red' for val in sorted_importance_disc]
        plt.barh(sorted_features_disc[::-1], sorted_importance_disc[::-1], color=colors_disc[::-1])
        plt.xlabel('Importance Score')
        plt.title(f'Discriminator IG Feature Importance - Step {idx+1} - Action {action}')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f'discriminator_ig_feature_importance_step_{idx+1}.png'))
        plt.close()

        # Optionally, visualize attributions for each channel (Policy Network - Integrated Gradients)
        num_channels = attributions_ig_list[idx].shape[0]
        for i in range(num_channels):
            plt.figure(figsize=(6, 4))
            plt.imshow(attributions_ig_list[idx][i], cmap='bwr', vmin=-np.max(np.abs(attributions_ig_list[idx][i])), vmax=np.max(np.abs(attributions_ig_list[idx][i])))
            plt.title(f'Policy IG - {feature_names[i]} - Step {idx+1} - Action {action}')
            plt.colorbar()
            plt.savefig(
                os.path.join(output_dir, f'policy_ig_channel_{i}_step_{idx+1}.png')
            )
            plt.close()

    # Plot the rewards over the trajectory
    plt.figure(figsize=(6, 4))
    plt.plot(range(1, len(rewards_list) + 1), rewards_list, marker='o')
    plt.title('Rewards over Trajectory Steps')
    plt.xlabel('Step')
    plt.ylabel('Reward')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'rewards_over_trajectory.png'))
    plt.close()

    print("Interpretation complete. Results saved in the 'output_img_step_by_step' directory.")

if __name__ == "__main__":
    interpret_model()
