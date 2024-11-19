import pandas as pd
from collections import defaultdict
import torch
import numpy as np
import csv
import os

# Function to analyze segments in the dataset
def analyze_segments(data_p, segment_length=3):
    # Read the dataset
    df = pd.read_csv(data_p)  # Adjust separator if needed

    # Ensure 'path' is a list of link numbers
    df['path'] = df['path'].apply(lambda x: x.split('_'))

    # Total number of trajectories
    total_trajectories = len(df)

    # Dictionary to hold segment information
    segment_info = defaultdict(lambda: {
        'start_link': None,
        'end_link': None,
        'trajectory_ids': set(),
        'origins': set(),
        'destinations': set(),
        'count': 0,
        'unique_trajectories': set()
    })

    # Iterate over each trajectory
    for idx, row in df.iterrows():
        ori = row['ori']
        des = row['des']
        path = row['path']
        trajectory_id = idx  # Use the DataFrame index as a unique trajectory ID
        trajectory_str = '_'.join(path)

        # Extract segments from the path
        for i in range(len(path) - segment_length + 1):
            segment = tuple(path[i:i + segment_length])
            start_link = segment[0]
            end_link = segment[-1]

            # Update segment information
            info = segment_info[segment]
            info['start_link'] = start_link
            info['end_link'] = end_link
            info['trajectory_ids'].add(trajectory_id)
            info['origins'].add(ori)
            info['destinations'].add(des)
            info['count'] += 1
            info['unique_trajectories'].add(trajectory_str)

    # Prepare data for analysis
    segment_analysis = []
    for segment, info in segment_info.items():
        num_unique_trajectories = len(info['unique_trajectories'])
        frequency = info['count'] / total_trajectories
        segment_analysis.append({
            'segment': '_'.join(segment),
            'start_link': info['start_link'],
            'end_link': info['end_link'],
            'trajectory_count': info['count'],
            'frequency': frequency,
            'num_unique_trajectories': num_unique_trajectories,
            'origins': list(info['origins']),
            'destinations': list(info['destinations'])
        })

    # Create a DataFrame for segment analysis
    df_segment_analysis = pd.DataFrame(segment_analysis)

    # Save the segment analysis to a CSV file
    df_segment_analysis.to_csv('./eva/segment_analysis_dataset.csv', index=False)
    print("Segment analysis for dataset saved to './eva/segment_analysis_dataset.csv'")

    return df_segment_analysis

# Function to analyze segments in generated trajectories
def analyze_segments_generated(generated_trajs, total_trajectories, segment_length=2):
    # Convert the list of generated trajectories into a DataFrame
    df_generated = pd.DataFrame(generated_trajs)

    # Ensure 'generated_trajectory' is a list of link numbers
    df_generated['generated_trajectory'] = df_generated['generated_trajectory'].apply(lambda x: x.split('_'))

    # Dictionary to hold segment information
    segment_info = defaultdict(lambda: {
        'start_link': None,
        'end_link': None,
        'trajectory_ids': set(),
        'origins': set(),
        'destinations': set(),
        'count': 0,
        'unique_trajectories': set()
    })

    # Iterate over each generated trajectory
    for idx, row in df_generated.iterrows():
        ori = row['origin']
        des = row['destination']
        path = row['generated_trajectory']
        trajectory_id = idx  # Use the DataFrame index as a unique trajectory ID
        trajectory_str = '_'.join(path)

        # Extract segments from the path
        for i in range(len(path) - segment_length + 1):
            segment = tuple(path[i:i + segment_length])
            start_link = segment[0]
            end_link = segment[-1]

            # Update segment information
            info = segment_info[segment]
            info['start_link'] = start_link
            info['end_link'] = end_link
            info['trajectory_ids'].add(trajectory_id)
            info['origins'].add(ori)
            info['destinations'].add(des)
            info['count'] += 1
            info['unique_trajectories'].add(trajectory_str)

    # Prepare data for analysis
    segment_analysis = []
    for segment, info in segment_info.items():
        num_unique_trajectories = len(info['unique_trajectories'])
        frequency = info['count'] / total_trajectories
        segment_analysis.append({
            'segment': '_'.join(segment),
            'start_link': info['start_link'],
            'end_link': info['end_link'],
            'trajectory_count': info['count'],
            'frequency': frequency,
            'num_unique_trajectories': num_unique_trajectories,
            'origins': list(info['origins']),
            'destinations': list(info['destinations'])
        })

    # Create a DataFrame for segment analysis
    df_segment_analysis = pd.DataFrame(segment_analysis)

    # Save the segment analysis to a CSV file
    df_segment_analysis.to_csv('./eva/segment_analysis_generated.csv', index=False)
    print("Segment analysis for generated data saved to './eva/segment_analysis_generated.csv'")

    return df_segment_analysis

# Function to generate trajectories matching the total quantity per OD
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

    # Analyze the dataset based on shared segments
    df_segment_analysis = analyze_segments(data_p, segment_length=2)

    # Load model and environment
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

    # Generate trajectories matching the total quantity per OD
    # Since we're focusing on segments, we'll need to generate trajectories based on all unique trajectories in the dataset

    # First, extract unique trajectories from the dataset
    df_dataset = pd.read_csv(data_p)
    df_dataset['path'] = df_dataset['path'].astype(str)
    unique_trajectories = df_dataset['path'].unique()

    # Prepare a DataFrame with origins and destinations for each unique trajectory
    df_unique_trajs = df_dataset.drop_duplicates(subset=['path'])[['ori', 'des', 'path']]

    # Generate trajectories matching the unique trajectories
    generated_trajs = []
    for idx, row in df_unique_trajs.iterrows():
        ori = row['ori']
        des = row['des']
        path = row['path']
        total_count = df_dataset[df_dataset['path'] == path].shape[0]

        # Generate the same number of trajectories as the count of this unique trajectory
        for _ in range(total_count):
            # Reset environment for the current OD pair
            curr_ori, curr_des = env.reset(st=ori, des=des)
            sample_path = [str(curr_ori)]
            curr_state = curr_ori

            for _ in range(50):
                state_tensor = torch.tensor([curr_state], dtype=torch.long).to(device)
                des_tensor = torch.tensor([curr_des], dtype=torch.long).to(device)

                # Get action probabilities from the model
                with torch.no_grad():
                    action_probs = policy_net.get_action_prob(state_tensor, des_tensor).squeeze()
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
                'dataset_path': path,
                'dataset_total_count': total_count
            })

    # Save generated trajectories
    output_filename = "./eva/generated_trajectories_matched_segments.csv"
    save_generated_trajectories_matched(generated_trajs, output_filename)

    # Analyze generated trajectories based on shared segments
    total_generated_trajectories = len(generated_trajs)
    df_segment_analysis_generated = analyze_segments_generated(generated_trajs, total_generated_trajectories, segment_length=2)

if __name__ == "__main__":
    main()





# import pandas as pd
# from collections import defaultdict, Counter
# import torch
# import numpy as np
# import csv
# import os

# # Function to analyze the dataset and get OD pairs and their statistics
# def analyze_dataset(data_p):
#     # Read the dataset
#     df = pd.read_csv(data_p)  # Adjust separator if needed

#     # Ensure 'path' is a string
#     df['path'] = df['path'].astype(str)

#     # Group by OD pair
#     od_group = df.groupby(['ori', 'des'])

#     # Prepare data for analysis
#     od_analysis = []
#     total_trajectories = len(df)

#     for od, group in od_group:
#         ori, des = od
#         total_count = len(group)
#         frequency = total_count / total_trajectories
#         unique_paths = group['path'].unique()
#         num_unique_paths = len(unique_paths)

#         od_analysis.append({
#             'origin': ori,
#             'destination': des,
#             'total_count': total_count,
#             'frequency': frequency,
#             'num_unique_paths': num_unique_paths,
#             'unique_paths': list(unique_paths)
#         })

#     # Create a DataFrame for analysis
#     df_analysis = pd.DataFrame(od_analysis)

#     # Save the first file: All unique OD trajectories with counts and frequencies
#     df_analysis.to_csv('./eva/seg_analysis_all.csv', index=False)

#     # Save the second file: OD trajectories with more than one kind of route
#     df_multiple_routes = df_analysis[df_analysis['num_unique_paths'] > 1]
#     df_multiple_routes.to_csv('./eva/seg_analysis_multiple_routes.csv', index=False)

#     # Save the third file: OD trajectories with only one kind of route
#     df_single_route = df_analysis[df_analysis['num_unique_paths'] == 1]
#     df_single_route.to_csv('./eva/seg_analysis_single_route.csv', index=False)

#     print("Analysis files saved:")
#     print("1. seg_analysis_all.csv")
#     print("2. seg_analysis_multiple_routes.csv")
#     print("3. seg_analysis_single_route.csv")

#     return df_analysis, df_multiple_routes, df_single_route

# # Function to analyze segments in the dataset
# def analyze_segments(data_p, segment_length=2):
#     # Read the dataset
#     df = pd.read_csv(data_p)  # Adjust separator if needed

#     # Ensure 'path' is a list of link numbers
#     df['path'] = df['path'].apply(lambda x: x.split('_'))

#     # Total number of trajectories
#     total_trajectories = len(df)

#     # Dictionary to hold segment information
#     segment_info = defaultdict(lambda: {
#         'start_link': None,
#         'end_link': None,
#         'trajectory_ids': set(),
#         'origins': set(),
#         'destinations': set(),
#         'count': 0
#     })

#     # Iterate over each trajectory
#     for idx, row in df.iterrows():
#         ori = row['ori']
#         des = row['des']
#         path = row['path']
#         trajectory_id = idx  # Use the DataFrame index as a unique trajectory ID

#         # Extract segments from the path
#         for i in range(len(path) - segment_length + 1):
#             segment = tuple(path[i:i + segment_length])
#             start_link = segment[0]
#             end_link = segment[-1]

#             # Update segment information
#             info = segment_info[segment]
#             info['start_link'] = start_link
#             info['end_link'] = end_link
#             info['trajectory_ids'].add(trajectory_id)
#             info['origins'].add(ori)
#             info['destinations'].add(des)
#             info['count'] += 1

#     # Prepare data for analysis
#     segment_analysis = []
#     for segment, info in segment_info.items():
#         num_unique_trajectories = len(info['trajectory_ids'])
#         frequency = info['count'] / total_trajectories
#         segment_analysis.append({
#             'segment': '_'.join(segment),
#             'start_link': info['start_link'],
#             'end_link': info['end_link'],
#             'trajectory_count': info['count'],
#             'frequency': frequency,
#             'num_unique_trajectories': num_unique_trajectories,
#             'origins': list(info['origins']),
#             'destinations': list(info['destinations'])
#         })

#     # Create a DataFrame for segment analysis
#     df_segment_analysis = pd.DataFrame(segment_analysis)

#     # Save the segment analysis to a CSV file
#     df_segment_analysis.to_csv('./eva/segment_analysis_dataset.csv', index=False)
#     print("Segment analysis for dataset saved to 'segment_analysis_dataset.csv'")

#     return df_segment_analysis

# # Function to analyze segments in generated trajectories
# def analyze_segments_generated(generated_trajs, total_trajectories, segment_length=2):
#     # Convert the list of generated trajectories into a DataFrame
#     df_generated = pd.DataFrame(generated_trajs)

#     # Ensure 'generated_trajectory' is a list of link numbers
#     df_generated['generated_trajectory'] = df_generated['generated_trajectory'].apply(lambda x: x.split('_'))

#     # Dictionary to hold segment information
#     segment_info = defaultdict(lambda: {
#         'start_link': None,
#         'end_link': None,
#         'trajectory_ids': set(),
#         'origins': set(),
#         'destinations': set(),
#         'count': 0
#     })

#     # Iterate over each generated trajectory
#     for idx, row in df_generated.iterrows():
#         ori = row['origin']
#         des = row['destination']
#         path = row['generated_trajectory']
#         trajectory_id = idx  # Use the DataFrame index as a unique trajectory ID

#         # Extract segments from the path
#         for i in range(len(path) - segment_length + 1):
#             segment = tuple(path[i:i + segment_length])
#             start_link = segment[0]
#             end_link = segment[-1]

#             # Update segment information
#             info = segment_info[segment]
#             info['start_link'] = start_link
#             info['end_link'] = end_link
#             info['trajectory_ids'].add(trajectory_id)
#             info['origins'].add(ori)
#             info['destinations'].add(des)
#             info['count'] += 1

#     # Prepare data for analysis
#     segment_analysis = []
#     for segment, info in segment_info.items():
#         num_unique_trajectories = len(info['trajectory_ids'])
#         frequency = info['count'] / total_trajectories
#         segment_analysis.append({
#             'segment': '_'.join(segment),
#             'start_link': info['start_link'],
#             'end_link': info['end_link'],
#             'trajectory_count': info['count'],
#             'frequency': frequency,
#             'num_unique_trajectories': num_unique_trajectories,
#             'origins': list(info['origins']),
#             'destinations': list(info['destinations'])
#         })

#     # Create a DataFrame for segment analysis
#     df_segment_analysis = pd.DataFrame(segment_analysis)

#     # Save the segment analysis to a CSV file
#     df_segment_analysis.to_csv('./eva/segment_analysis_generated.csv', index=False)
#     print("Segment analysis for generated data saved to 'segment_analysis_generated.csv'")

#     return df_segment_analysis

# # Function to generate trajectories matching the total quantity per OD
# def generate_trajectories_matched(df_analysis, model, env, device, max_steps=50):
#     generated_trajs = []

#     for idx, row in df_analysis.iterrows():
#         ori = row['origin']
#         des = row['destination']
#         total_count = int(row['total_count'])
#         unique_paths = row['unique_paths']
#         frequency = row['frequency']

#         # Generate the same number of trajectories as in the dataset
#         for _ in range(total_count):
#             # Reset environment for the current OD pair
#             curr_ori, curr_des = env.reset(st=ori, des=des)
#             sample_path = [str(curr_ori)]
#             curr_state = curr_ori

#             for _ in range(max_steps):
#                 state_tensor = torch.tensor([curr_state], dtype=torch.long).to(device)
#                 des_tensor = torch.tensor([curr_des], dtype=torch.long).to(device)

#                 # Get action probabilities from the model
#                 with torch.no_grad():
#                     action_probs = model.get_action_prob(state_tensor, des_tensor).squeeze()
#                     action_probs_np = action_probs.cpu().numpy()

#                 # Sample an action based on probabilities
#                 action = np.random.choice(len(action_probs_np), p=action_probs_np)

#                 # Take a step in the environment
#                 next_state, reward, done = env.step(action)

#                 # Append the next state to the path
#                 if next_state != env.pad_idx:
#                     sample_path.append(str(next_state))
#                 curr_state = next_state

#                 if done:
#                     break

#             # Append the generated trajectory and associated information
#             generated_trajs.append({
#                 'origin': ori,
#                 'destination': des,
#                 'generated_trajectory': '_'.join(sample_path),
#                 'dataset_unique_paths': unique_paths,
#                 'dataset_total_count': total_count,
#                 'dataset_frequency': frequency
#             })

#     return generated_trajs

# # Function to save generated trajectories
# def save_generated_trajectories_matched(generated_trajs, filename):
#     # Convert the list of dictionaries to a DataFrame
#     df_generated = pd.DataFrame(generated_trajs)

#     # Save to CSV
#     df_generated.to_csv(filename, index=False)
#     print(f"Generated trajectories saved to {filename}")

# def main():
#     device = torch.device('cpu')  # or 'cuda' if using GPU

#     # Paths
#     data_p = "../data/base/cross_validation/train_CV0_size10000.csv"
#     model_path = "../trained_models/base/airl_CV0_size10000.pt"  # Adjust as necessary
#     edge_p = "../data/base/edge.txt"
#     network_p = "../data/base/transit.npy"
#     path_feature_p = "../data/base/feature_od.npy"

#     # Analyze the dataset to get df_analysis for generating trajectories
#     df_analysis, _, _ = analyze_dataset(data_p)

#     # Analyze the dataset based on shared segments
#     df_segment_analysis = analyze_segments(data_p, segment_length=2)

#     # Load model and environment
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

#     # Generate trajectories matching the total quantity per OD
#     generated_trajs = generate_trajectories_matched(df_analysis, policy_net, env, device, max_steps=50)

#     # Save generated trajectories
#     output_filename = "./eva/generated_trajectories_matched.csv"
#     save_generated_trajectories_matched(generated_trajs, output_filename)

#     # Analyze generated trajectories based on shared segments
#     total_generated_trajectories = len(generated_trajs)
#     df_segment_analysis_generated = analyze_segments_generated(generated_trajs, total_generated_trajectories, segment_length=2)

# if __name__ == "__main__":
#     main()
