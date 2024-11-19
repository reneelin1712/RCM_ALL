# Import necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import ast
from sklearn.cluster import KMeans

# Load the OD analysis with multiple routes
od_analysis = pd.read_csv('eva/od_analysis_multiple_routes.csv')

# Extract the relevant origin-destination pairs
od_pairs = od_analysis[['origin', 'destination']]

# Load the original dataset
df = pd.read_csv('eva/generated_trajectories_with_rewards.csv')

# Ensure numeric columns are properly typed
numeric_columns = ['origin', 'destination', 'trajectory_length', 'total_return',
                   'highest_reward_index', 'highest_reward_value', 'lowest_reward_index',
                   'lowest_reward_value', 'divergence_index', 'convergence_index']
for col in numeric_columns:
    df[col] = pd.to_numeric(df[col], errors='coerce')

# Filter the original dataset to include only matching OD pairs
filtered_df = df.merge(od_pairs, on=['origin', 'destination'], how='inner')

# Save the filtered dataset for reference
filtered_df.to_csv('filtered_trajectories_with_multiple_routes.csv', index=False)
print(f"Filtered dataset saved to 'filtered_trajectories_with_multiple_routes.csv'.")

# Perform reward analysis on the filtered dataset
# Convert 'rewards_per_step' from string representation to list
def parse_rewards(s):
    try:
        return ast.literal_eval(s)
    except:
        return []

filtered_df['rewards_per_step_list'] = filtered_df['rewards_per_step'].apply(parse_rewards)
filtered_df['trajectory_length'] = filtered_df['rewards_per_step_list'].apply(len)

# Handle NaN values in index columns
index_columns = ['highest_reward_index', 'lowest_reward_index', 'divergence_index', 'convergence_index']
for col in index_columns:
    filtered_df[col] = pd.to_numeric(filtered_df[col], errors='coerce')

# Calculate relative positions
filtered_df['relative_highest_reward_position'] = filtered_df['highest_reward_index'] / filtered_df['trajectory_length']
filtered_df['relative_lowest_reward_position'] = filtered_df['lowest_reward_index'] / filtered_df['trajectory_length']
filtered_df['relative_divergence_position'] = filtered_df['divergence_index'] / filtered_df['trajectory_length']
filtered_df['relative_convergence_position'] = filtered_df['convergence_index'] / filtered_df['trajectory_length']

# Plot histograms of relative positions
plt.figure(figsize=(12, 6))
plt.hist(filtered_df['relative_highest_reward_position'].dropna(), bins=20, alpha=0.5, label='Highest Reward Position')
plt.hist(filtered_df['relative_lowest_reward_position'].dropna(), bins=20, alpha=0.5, label='Lowest Reward Position')
plt.title('Distribution of Relative Positions of Highest and Lowest Rewards')
plt.xlabel('Relative Position in Trajectory')
plt.ylabel('Frequency')
plt.legend()
plt.show()

# Function to compute trend correlation
def compute_trend(rewards):
    if len(rewards) < 2:
        return np.nan
    steps = np.arange(len(rewards))
    corr = np.corrcoef(steps, rewards)[0, 1]
    return corr

filtered_df['reward_trend_correlation'] = filtered_df['rewards_per_step_list'].apply(compute_trend)

# Classify trends
def classify_trend(corr):
    if pd.isnull(corr):
        return 'undefined'
    elif corr > 0.5:
        return 'increasing'
    elif corr < -0.5:
        return 'decreasing'
    else:
        return 'random'

filtered_df['reward_trend'] = filtered_df['reward_trend_correlation'].apply(classify_trend)

# Count the number of trajectories in each category
trend_counts = filtered_df['reward_trend'].value_counts()
print("\nReward Trend Counts:")
print(trend_counts)

# Visualize the trends
plt.figure(figsize=(6, 6))
trend_counts.plot(kind='pie', autopct='%1.1f%%', startangle=90)
plt.title('Distribution of Reward Trends')
plt.ylabel('')
plt.show()

# Prepare data for clustering
def pad_sequences_custom(sequences, maxlen=None, padding='post', value=0.0):
    if not maxlen:
        maxlen = max(len(seq) for seq in sequences)
    padded_sequences = []
    for seq in sequences:
        if len(seq) < maxlen:
            if padding == 'post':
                seq = seq + [value] * (maxlen - len(seq))
            elif padding == 'pre':
                seq = [value] * (maxlen - len(seq)) + seq
        else:
            seq = seq[:maxlen]
        padded_sequences.append(seq)
    return np.array(padded_sequences)

rewards_list = filtered_df['rewards_per_step_list'].tolist()
padded_rewards = pad_sequences_custom(rewards_list, padding='post', value=0.0)

# Perform clustering
kmeans = KMeans(n_clusters=3, random_state=42)
clusters = kmeans.fit_predict(padded_rewards)

filtered_df['cluster'] = clusters

# Analyze clusters
for i in range(3):
    cluster_df = filtered_df[filtered_df['cluster'] == i]
    print(f"\nCluster {i} has {cluster_df.shape[0]} trajectories.")

# Plot reward trends for a sample of trajectories
sample_df = filtered_df.sample(n=5, random_state=42)

for idx, row in sample_df.iterrows():
    rewards = row['rewards_per_step_list']
    plt.figure()
    plt.plot(range(1, len(rewards) + 1), rewards, marker='o')
    plt.title(f"Trajectory {idx} Reward Trend (OD: {row['origin']} -> {row['destination']})")
    plt.xlabel('Step')
    plt.ylabel('Reward')
    plt.show()

# Create a DataFrame of padded rewards
reward_matrix = pd.DataFrame(padded_rewards)
reward_matrix['cluster'] = filtered_df['cluster']

# Average rewards per step per cluster
avg_rewards_per_step = reward_matrix.groupby('cluster').mean()

# Plot average rewards per step per cluster
plt.figure(figsize=(12, 6))
sns.heatmap(avg_rewards_per_step, annot=True, cmap='viridis')
plt.title('Average Rewards per Step per Cluster')
plt.xlabel('Step')
plt.ylabel('Cluster')
plt.show()

# Save the DataFrame with analyses
filtered_df.to_csv('filtered_generated_trajectories_with_rewards_analysis.csv', index=False)
print("Filtered and analyzed DataFrame saved to 'filtered_generated_trajectories_with_rewards_analysis.csv'.")

# # Import necessary libraries
# import pandas as pd
# import numpy as np
# import ast

# # Load the generated trajectories with rewards
# df = pd.read_csv('eva/generated_trajectories_with_rewards.csv')

# # Load the dataset paths and counts
# df_per_path = pd.read_csv('eva/od_analysis_per_path.csv')

# # Ensure numeric columns are properly typed in df
# numeric_columns = ['origin', 'destination', 'trajectory_length', 'total_return',
#                    'highest_reward_index', 'highest_reward_value', 'lowest_reward_index',
#                    'lowest_reward_value', 'divergence_index', 'convergence_index']
# for col in numeric_columns:
#     df[col] = pd.to_numeric(df[col], errors='coerce')

# # Convert 'rewards_per_step' from string representation to list
# def parse_rewards(s):
#     try:
#         return ast.literal_eval(s)
#     except:
#         return []

# df['rewards_per_step_list'] = df['rewards_per_step'].apply(parse_rewards)
# df['trajectory_length'] = df['rewards_per_step_list'].apply(len)

# # Handle NaN values in index columns
# index_columns = ['highest_reward_index', 'lowest_reward_index', 'divergence_index', 'convergence_index']
# for col in index_columns:
#     df[col] = pd.to_numeric(df[col], errors='coerce')

# # Create a dictionary mapping OD pairs to set of dataset_paths
# od_to_paths = {}
# for idx, row in df_per_path.iterrows():
#     ori = row['origin']
#     des = row['destination']
#     path = row['path']
#     od_pair = (ori, des)
#     if od_pair not in od_to_paths:
#         od_to_paths[od_pair] = set()
#     od_to_paths[od_pair].add(path)

# # Update 'dataset_path' in df: set to NaN if generated_trajectory doesn't match any dataset_path
# def update_dataset_path(row):
#     ori = row['origin']
#     des = row['destination']
#     generated_traj = row['generated_trajectory']
#     od_pair = (ori, des)
#     if od_pair in od_to_paths:
#         if generated_traj in od_to_paths[od_pair]:
#             return row['dataset_path']  # Keep dataset_path
#         else:
#             return np.nan  # Set to NaN
#     else:
#         return np.nan  # Set to NaN

# df['dataset_path'] = df.apply(update_dataset_path, axis=1)

# # Merge df with df_per_path to get 'path_count' and other info
# df_merged = pd.merge(df, df_per_path[['origin', 'destination', 'path', 'path_count']],
#                      how='left',
#                      left_on=['origin', 'destination', 'dataset_path'],
#                      right_on=['origin', 'destination', 'path'])

# # Check if 'path_count' exists
# if 'path_count' not in df_merged.columns:
#     print("Error: 'path_count' not found in merged DataFrame.")
#     # Handle error or exit
#     df_merged['path_count'] = np.nan  # Add 'path_count' column with NaN values

# # Compute average reward per trajectory
# df_merged['average_reward'] = df_merged['rewards_per_step_list'].apply(lambda x: np.mean(x) if len(x) > 0 else np.nan)

# # Compute standard deviation of rewards per trajectory
# df_merged['reward_std'] = df_merged['rewards_per_step_list'].apply(lambda x: np.std(x) if len(x) > 0 else np.nan)

# # Compute differences between divergence/convergence rewards and average reward
# df_merged['divergence_reward_diff'] = df_merged['divergence_reward'] - df_merged['average_reward']
# df_merged['convergence_reward_diff'] = df_merged['convergence_reward'] - df_merged['average_reward']

# # Compute z-scores for divergence and convergence rewards
# df_merged['divergence_reward_zscore'] = df_merged.apply(
#     lambda row: (row['divergence_reward'] - row['average_reward']) / row['reward_std'] if pd.notnull(row['divergence_reward']) and row['reward_std'] != 0 else np.nan,
#     axis=1)
# df_merged['convergence_reward_zscore'] = df_merged.apply(
#     lambda row: (row['convergence_reward'] - row['average_reward']) / row['reward_std'] if pd.notnull(row['convergence_reward']) and row['reward_std'] != 0 else np.nan,
#     axis=1)

# # Group by OD pair for analysis
# od_groups = df_merged.groupby(['origin', 'destination'])

# for od_pair, group in od_groups:
#     ori, des = od_pair
#     print(f"\nOD Pair: {ori} -> {des}")
    
#     # List dataset_paths and their counts
#     dataset_paths = group['dataset_path'].dropna().unique()
#     print("Dataset Paths and Counts:")
#     for path in dataset_paths:
#         path_count_series = group[group['dataset_path'] == path]['path_count']
#         if not path_count_series.empty and not pd.isnull(path_count_series.iloc[0]):
#             path_count = int(path_count_series.iloc[0])
#         else:
#             path_count = 'Unknown'
#         print(f"  Path: {path}, Count: {path_count}")
    
#     # List generated trajectories
#     print("Generated Trajectories:")
#     for idx, row in group.iterrows():
#         gen_traj = row['generated_trajectory']
#         dataset_path = row['dataset_path']
#         if pd.isnull(dataset_path):
#             match_status = 'No match'
#         else:
#             match_status = 'Matches Dataset Path'
#         print(f"  Generated Trajectory ({match_status}): {gen_traj}")
#         # Print rewards statistics
#         print(f"    Total Return: {row['total_return']:.4f}")
#         print(f"    Average Reward: {row['average_reward']:.4f}")
#         print(f"    Divergence Reward Difference: {row['divergence_reward_diff']}")
#         print(f"    Convergence Reward Difference: {row['convergence_reward_diff']}")
#         print(f"    Divergence Reward Z-Score: {row['divergence_reward_zscore']}")
#         print(f"    Convergence Reward Z-Score: {row['convergence_reward_zscore']}")
    
#     # Additional analysis: Compare rewards at divergence/convergence steps
#     # For trajectories with the same OD but different paths
#     if group['generated_trajectory'].nunique() > 1:
#         print("\nComparison of Rewards at Divergence and Convergence Steps:")
#         for idx, row in group.iterrows():
#             gen_traj = row['generated_trajectory']
#             print(f"  Trajectory: {gen_traj}")
#             print(f"    Divergence Reward: {row['divergence_reward']}")
#             print(f"    Convergence Reward: {row['convergence_reward']}")
#             print(f"    Divergence Reward Z-Score: {row['divergence_reward_zscore']}")
#             print(f"    Convergence Reward Z-Score: {row['convergence_reward_zscore']}")
