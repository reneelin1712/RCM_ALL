import pandas as pd
from collections import Counter
import torch
import numpy as np
import csv
import os

# Function to analyze the dataset and select OD pairs
def select_od_pairs(data_p, num_pairs=10):
    df = pd.read_csv(data_p)  # Adjust separator if needed
    od_pairs = list(zip(df['ori'], df['des']))
    od_counter = Counter(od_pairs)
    most_common_od = od_counter.most_common(num_pairs)
    print("Most common OD pairs:")
    for od, count in most_common_od:
        print(f"OD pair {od} appears {count} times")
    selected_od_pairs = [od for od, count in most_common_od]
    return selected_od_pairs



def analyze_dataset(data_p):
    # Read the dataset
    df = pd.read_csv(data_p)  # Adjust separator if needed
    
    # Ensure 'path' is a string
    df['path'] = df['path'].astype(str)
    
    # Group by OD pair
    od_group = df.groupby(['ori', 'des'])
    
    # Prepare data for analysis
    od_analysis = []
    total_trajectories = len(df)
    
    for od, group in od_group:
        ori, des = od
        total_count = len(group)
        frequency = total_count / total_trajectories
        unique_paths = group['path'].unique()
        num_unique_paths = len(unique_paths)
        
        od_analysis.append({
            'origin': ori,
            'destination': des,
            'total_count': total_count,
            'frequency': frequency,
            'num_unique_paths': num_unique_paths,
            'unique_paths': list(unique_paths)
        })
    
    # Create a DataFrame for analysis
    df_analysis = pd.DataFrame(od_analysis)
    
    # Save the first file: All unique OD trajectories with counts and frequencies
    df_analysis.to_csv('./eva/od_analysis_all.csv', index=False)
    
    # Save the second file: OD trajectories with more than one kind of route
    df_multiple_routes = df_analysis[df_analysis['num_unique_paths'] > 1]
    df_multiple_routes.to_csv('./eva/od_analysis_multiple_routes.csv', index=False)
    
    # Save the third file: OD trajectories with only one kind of route
    df_single_route = df_analysis[df_analysis['num_unique_paths'] == 1]
    df_single_route.to_csv('./eva/od_analysis_single_route.csv', index=False)
    
    print("Analysis files saved:")
    print("1. od_analysis_all.csv")
    print("2. od_analysis_multiple_routes.csv")
    print("3. od_analysis_single_route.csv")
    
    return df_analysis, df_multiple_routes, df_single_route


# Function to generate trajectories
def generate_trajectories(od_pairs, model, env, device, max_steps=50):
    generated_trajs = []
    for idx, (ori, des) in enumerate(od_pairs):
        # Reset environment for the current origin-destination pair
        curr_ori, curr_des = env.reset(st=ori, des=des)
        sample_path = [str(curr_ori)]
        curr_state = curr_ori

        for _ in range(max_steps):
            state_tensor = torch.tensor([curr_state], dtype=torch.long).to(device)
            des_tensor = torch.tensor([curr_des], dtype=torch.long).to(device)

            # Get action probabilities from the model for the current state and destination
            with torch.no_grad():
                action_probs = model.get_action_prob(state_tensor, des_tensor).squeeze()
                action_probs_np = action_probs.cpu().numpy()

            # Choose the most likely action based on the model
            action = np.argmax(action_probs_np)

            # Take a step in the environment
            next_state, reward, done = env.step(action)

            # Append the next state to the path
            if next_state != env.pad_idx:
                sample_path.append(str(next_state))
            curr_state = next_state

            if done:
                break

        # Append the generated path for the current OD pair
        generated_trajs.append(sample_path)

    return generated_trajs

def generate_trajectories_matched(df_analysis, model, env, device, max_steps=50):
    generated_trajs = []
    
    for idx, row in df_analysis.iterrows():
        ori = row['origin']
        des = row['destination']
        total_count = int(row['total_count'])
        unique_paths = row['unique_paths']
        frequency = row['frequency']
        
        # Generate the same number of trajectories as in the dataset
        for _ in range(total_count):
            # Reset environment for the current OD pair
            curr_ori, curr_des = env.reset(st=ori, des=des)
            sample_path = [str(curr_ori)]
            curr_state = curr_ori
            
            for _ in range(max_steps):
                state_tensor = torch.tensor([curr_state], dtype=torch.long).to(device)
                des_tensor = torch.tensor([curr_des], dtype=torch.long).to(device)
                
                # Get action probabilities from the model
                with torch.no_grad():
                    action_probs = model.get_action_prob(state_tensor, des_tensor).squeeze()
                    action_probs_np = action_probs.cpu().numpy()
                    
                # Sample an action based on probabilities
                action = np.random.choice(len(action_probs_np), p=action_probs_np)
                
                # Take a step in the environment
                next_state, reward, done = env.step(action)
                
                # Append the next state to the path
                if next_state != env.pad_idx:
                    sample_path.append(str(next_state))
                curr_state = next_state
                
                if done:
                    break
            
            # Append the generated trajectory and associated information
            generated_trajs.append({
                'origin': ori,
                'destination': des,
                'generated_trajectory': '_'.join(sample_path),
                'dataset_unique_paths': unique_paths,
                'dataset_total_count': total_count,
                'dataset_frequency': frequency
            })
    
    return generated_trajs

# Function to save generated trajectories
def save_generated_trajectories(od_pairs, generated_trajs, filename):
    with open(filename, 'w', newline='') as csvfile:
        csv_writer = csv.writer(csvfile)
        csv_writer.writerow(["Origin", "Destination", "Generated Trajectory"])
        for (ori, des), traj in zip(od_pairs, generated_trajs):
            traj_str = '_'.join(traj)
            csv_writer.writerow([ori, des, traj_str])

def save_generated_trajectories_matched(generated_trajs, filename):
    # Convert the list of dictionaries to a DataFrame
    df_generated = pd.DataFrame(generated_trajs)
    
    # Save to CSV
    df_generated.to_csv(filename, index=False)
    print(f"Generated trajectories saved to {filename}")


def main():
    device = torch.device('cpu')  # or 'cuda' if using GPU

    # Paths
    data_p = "../data/base/cross_validation/train_CV0_size10000.csv"
    model_path = "../trained_models/base/airl_CV0_size10000.pt"  # Adjust as necessary
    edge_p = "../data/base/edge.txt"
    network_p = "../data/base/transit.npy"
    path_feature_p = "../data/base/feature_od.npy"

    # Step 1: Analyze the dataset and save analysis files
    df_analysis, df_multiple_routes, df_single_route = analyze_dataset(data_p)

    # Step 2: Load model and environment
    from network_env import RoadWorld
    from utils.load_data import ini_od_dist, load_path_feature, load_link_feature, minmax_normalization
    from model.policy import PolicyCNN
    from model.value import ValueCNN
    from model.discriminator import DiscriminatorAIRLCNN

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
    def load_model(model_path, device, env, path_feature_pad, edge_feature_pad):
        gamma = 0.95  # Adjust gamma as needed
        policy_net = PolicyCNN(env.n_actions, env.policy_mask, env.state_action,
                               path_feature_pad, edge_feature_pad,
                               path_feature_pad.shape[-1] + edge_feature_pad.shape[-1] + 1,
                               env.pad_idx).to(device)
        value_net = ValueCNN(path_feature_pad, edge_feature_pad,
                             path_feature_pad.shape[-1] + edge_feature_pad.shape[-1]).to(device)
        discrim_net = DiscriminatorAIRLCNN(env.n_actions, gamma, env.policy_mask,
                                           env.state_action, path_feature_pad, edge_feature_pad,
                                           path_feature_pad.shape[-1] + edge_feature_pad.shape[-1] + 1,
                                           path_feature_pad.shape[-1] + edge_feature_pad.shape[-1],
                                           env.pad_idx).to(device)

        model_dict = torch.load(model_path, map_location=device)
        policy_net.load_state_dict(model_dict['Policy'])
        value_net.load_state_dict(model_dict['Value'])
        discrim_net.load_state_dict(model_dict['Discrim'])

        return policy_net, value_net, discrim_net

    policy_net, value_net, discrim_net = load_model(model_path, device, env, path_feature_pad, edge_feature_pad)

    # Step 3: Generate trajectories matching the total quantity per OD
    generated_trajs = generate_trajectories_matched(df_analysis, policy_net, env, device, max_steps=50)

    # Step 4: Save generated trajectories
    output_filename = "./eva/generated_trajectories_matched.csv"
    save_generated_trajectories_matched(generated_trajs, output_filename)

if __name__ == "__main__":
    main()



# # Main function
# def main():
#     device = torch.device('cpu')  # or 'cuda' if using GPU

#     # Paths
#     data_p = "../data/base/cross_validation/train_CV0_size10000.csv"
#     model_path = "../trained_models/base/airl_CV0_size10000.pt"  # Adjust as necessary
#     edge_p = "../data/base/edge.txt"
#     network_p = "../data/base/transit.npy"
#     path_feature_p = "../data/base/feature_od.npy"

#     # Number of OD pairs to select
#     num_od_pairs = 10

#     # Step 1: Select OD pairs
#     selected_od_pairs = select_od_pairs(data_p, num_od_pairs)

#     # Step 2: Load model and environment
#     from network_env import RoadWorld
#     from utils.load_data import ini_od_dist, load_path_feature, load_link_feature, minmax_normalization
#     from model.policy import PolicyCNN
#     from model.value import ValueCNN
#     from model.discriminator import DiscriminatorAIRLCNN

#     # Initialize environment
#     od_list, od_dist = ini_od_dist(data_p)
#     env = RoadWorld(network_p, edge_p, pre_reset=(od_list, od_dist))

#     # Load features and normalize
#     path_feature, path_max, path_min = load_path_feature(path_feature_p)
#     edge_feature, link_max, link_min = load_link_feature(edge_p)
#     path_feature = minmax_normalization(path_feature, path_max, path_min)
#     path_feature_pad = np.zeros((env.n_states, env.n_states, path_feature.shape[2]))
#     path_feature_pad[:path_feature.shape[0], :path_feature.shape[1], :] = path_feature
#     edge_feature = minmax_normalization(edge_feature, link_max, link_min)
#     edge_feature_pad = np.zeros((env.n_states, edge_feature.shape[1]))
#     edge_feature_pad[:edge_feature.shape[0], :] = edge_feature

#     # Load the model
#     def load_model(model_path, device, env, path_feature_pad, edge_feature_pad):
#         gamma = 0.95  # Adjust gamma as needed
#         policy_net = PolicyCNN(env.n_actions, env.policy_mask, env.state_action,
#                                path_feature_pad, edge_feature_pad,
#                                path_feature_pad.shape[-1] + edge_feature_pad.shape[-1] + 1,
#                                env.pad_idx).to(device)
#         value_net = ValueCNN(path_feature_pad, edge_feature_pad,
#                              path_feature_pad.shape[-1] + edge_feature_pad.shape[-1]).to(device)
#         discrim_net = DiscriminatorAIRLCNN(env.n_actions, gamma, env.policy_mask,
#                                            env.state_action, path_feature_pad, edge_feature_pad,
#                                            path_feature_pad.shape[-1] + edge_feature_pad.shape[-1] + 1,
#                                            path_feature_pad.shape[-1] + edge_feature_pad.shape[-1],
#                                            env.pad_idx).to(device)

#         model_dict = torch.load(model_path, map_location=device)
#         policy_net.load_state_dict(model_dict['Policy'])
#         value_net.load_state_dict(model_dict['Value'])
#         discrim_net.load_state_dict(model_dict['Discrim'])

#         return policy_net, value_net, discrim_net

#     policy_net, value_net, discrim_net = load_model(model_path, device, env, path_feature_pad, edge_feature_pad)

#     # Step 3: Generate trajectories
#     generated_trajs = generate_trajectories(selected_od_pairs, policy_net, env, device, max_steps=50)

#     # Step 4: Save generated trajectories
#     output_filename = "./eva/generated_trajectories.csv"
#     save_generated_trajectories(selected_od_pairs, generated_trajs, output_filename)
#     print(f"Generated trajectories saved to {output_filename}")

# if __name__ == "__main__":
#     main()
