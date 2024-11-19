# Import necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import ast
from sklearn.cluster import KMeans
import seaborn as sns

# Load the CSV file
df = pd.read_csv('eva/generated_trajectories_with_rewards.csv')

# Ensure numeric columns are properly typed
numeric_columns = ['origin', 'destination', 'trajectory_length', 'total_return',
                   'highest_reward_index', 'highest_reward_value', 'lowest_reward_index',
                   'lowest_reward_value', 'divergence_index', 'convergence_index']
for col in numeric_columns:
    df[col] = pd.to_numeric(df[col], errors='coerce')

# Convert 'rewards_per_step' from string representation to list
def parse_rewards(s):
    try:
        return ast.literal_eval(s)
    except:
        return []

df['rewards_per_step_list'] = df['rewards_per_step'].apply(parse_rewards)
df['trajectory_length'] = df['rewards_per_step_list'].apply(len)

# Handle NaN values in index columns
index_columns = ['highest_reward_index', 'lowest_reward_index', 'divergence_index', 'convergence_index']
for col in index_columns:
    df[col] = pd.to_numeric(df[col], errors='coerce')

# Calculate relative positions
df['relative_highest_reward_position'] = df['highest_reward_index'] / df['trajectory_length']
df['relative_lowest_reward_position'] = df['lowest_reward_index'] / df['trajectory_length']
df['relative_divergence_position'] = df['divergence_index'] / df['trajectory_length']
df['relative_convergence_position'] = df['convergence_index'] / df['trajectory_length']

# Count trajectories where highest reward is at the last step
highest_reward_at_last_step = df[df['highest_reward_index'] == df['trajectory_length']].shape[0]
total_trajectories = df.shape[0]

print(f"Total number of trajectories: {total_trajectories}")
print(f"Number where highest reward is at the last step: {highest_reward_at_last_step}")
print(f"Percentage: {highest_reward_at_last_step / total_trajectories * 100:.2f}%")

# Handle NaN values
df['divergence_index'] = df['divergence_index'].fillna(-1).astype(int)
df['convergence_index'] = df['convergence_index'].fillna(-1).astype(int)

# Highest reward at divergence step
highest_reward_at_divergence = df[df['highest_reward_index'] == df['divergence_index']]
num_highest_at_divergence = highest_reward_at_divergence.shape[0]

# Highest reward at convergence step
highest_reward_at_convergence = df[df['highest_reward_index'] == df['convergence_index']]
num_highest_at_convergence = highest_reward_at_convergence.shape[0]

print(f"Number where highest reward is at divergence step: {num_highest_at_divergence}")
print(f"Number where highest reward is at convergence step: {num_highest_at_convergence}")

# Plot histograms of relative positions
plt.figure(figsize=(12, 6))
plt.hist(df['relative_highest_reward_position'].dropna(), bins=20, alpha=0.5, label='Highest Reward Position')
plt.hist(df['relative_lowest_reward_position'].dropna(), bins=20, alpha=0.5, label='Lowest Reward Position')
plt.title('Distribution of Relative Positions of Highest and Lowest Rewards')
plt.xlabel('Relative Position in Trajectory')
plt.ylabel('Frequency')
plt.legend()
plt.show()

# Group by OD pair
od_groups = df.groupby(['origin', 'destination'])

# Analyze positions within each group
for (ori, des), group in od_groups:
    if group.shape[0] > 1:
        print(f"\nOD Pair ({ori} -> {des}) with {group.shape[0]} trajectories:")
        print("Highest Reward Indices:", group['highest_reward_index'].tolist())
        print("Lowest Reward Indices:", group['lowest_reward_index'].tolist())
        print("Divergence Indices:", group['divergence_index'].tolist())
        print("Convergence Indices:", group['convergence_index'].tolist())

# Function to compute trend correlation
def compute_trend(rewards):
    if len(rewards) < 2:
        return np.nan
    steps = np.arange(len(rewards))
    corr = np.corrcoef(steps, rewards)[0, 1]
    return corr

df['reward_trend_correlation'] = df['rewards_per_step_list'].apply(compute_trend)

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

df['reward_trend'] = df['reward_trend_correlation'].apply(classify_trend)

# Count the number of trajectories in each category
trend_counts = df['reward_trend'].value_counts()
print("\nReward Trend Counts:")
print(trend_counts)

# Visualize the trends
plt.figure(figsize=(6, 6))
trend_counts.plot(kind='pie', autopct='%1.1f%%', startangle=90)
plt.title('Distribution of Reward Trends')
plt.ylabel('')
plt.show()

# OD pairs with multiple trajectories
od_counts = df.groupby(['origin', 'destination']).size()
od_pairs_multiple = od_counts[od_counts > 1].index.tolist()

for od_pair in od_pairs_multiple:
    ori, des = od_pair
    od_df = df[(df['origin'] == ori) & (df['destination'] == des)]
    num_trajs = od_df.shape[0]
    print(f"\nOD Pair ({ori} -> {des}) with {num_trajs} trajectories:")
    
    # Collect indices
    highest_indices = od_df['highest_reward_index'].tolist()
    lowest_indices = od_df['lowest_reward_index'].tolist()
    divergence_indices = od_df['divergence_index'].tolist()
    convergence_indices = od_df['convergence_index'].tolist()
    
    # Display the indices
    print("Highest Reward Indices:", highest_indices)
    print("Lowest Reward Indices:", lowest_indices)
    print("Divergence Indices:", divergence_indices)
    print("Convergence Indices:", convergence_indices)
    
    # Check for consistency
    if len(set(highest_indices)) == 1:
        print("Highest reward positions are consistent.")
    else:
        print("Highest reward positions vary.")
    
    if len(set(divergence_indices)) == 1 and divergence_indices[0] != -1:
        print("Divergence positions are consistent.")
    else:
        print("Divergence positions vary.")

# Rewards at divergence and convergence steps
def get_reward_at_index(rewards, index):
    if pd.isnull(index) or index <= 0 or index > len(rewards):
        return np.nan
    return rewards[int(index)-1]

df['reward_at_divergence'] = df.apply(lambda row: get_reward_at_index(row['rewards_per_step_list'], row['divergence_index']), axis=1)
df['reward_at_convergence'] = df.apply(lambda row: get_reward_at_index(row['rewards_per_step_list'], row['convergence_index']), axis=1)

# Calculate average rewards
avg_reward_divergence = df['reward_at_divergence'].mean()
avg_reward_convergence = df['reward_at_convergence'].mean()

print(f"\nAverage Reward at Divergence Steps: {avg_reward_divergence:.4f}")
print(f"Average Reward at Convergence Steps: {avg_reward_convergence:.4f}")

# Compute standard deviation of rewards per trajectory
df['reward_std'] = df['rewards_per_step_list'].apply(lambda x: np.std(x) if len(x) > 0 else np.nan)

# Analyze the distribution
plt.figure(figsize=(8, 6))
plt.hist(df['reward_std'].dropna(), bins=30)
plt.title('Distribution of Reward Standard Deviation in Trajectories')
plt.xlabel('Standard Deviation of Rewards')
plt.ylabel('Frequency')
plt.show()

# Total Return Comparison for Same OD Pairs
for od_pair in od_pairs_multiple:
    ori, des = od_pair
    od_df = df[(df['origin'] == ori) & (df['destination'] == des)]
    print(f"\nOD Pair ({ori} -> {des}):")
    print("Total Returns:", od_df['total_return'].tolist())
    avg_return = od_df['total_return'].mean()
    print(f"Average Total Return: {avg_return:.4f}")

# Function to pad sequences to the same length
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

# Prepare data for clustering
rewards_list = df['rewards_per_step_list'].tolist()
# Pad sequences to the same length
padded_rewards = pad_sequences_custom(rewards_list, padding='post', value=0.0)

# Perform clustering
kmeans = KMeans(n_clusters=3, random_state=42)
clusters = kmeans.fit_predict(padded_rewards)

df['cluster'] = clusters

# Analyze clusters
for i in range(3):
    cluster_df = df[df['cluster'] == i]
    print(f"\nCluster {i} has {cluster_df.shape[0]} trajectories.")
    # Further analysis per cluster can be added here

# Plot reward trends for a sample of trajectories
sample_df = df.sample(n=5, random_state=42)

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
reward_matrix['cluster'] = df['cluster']

# Average rewards per step per cluster
avg_rewards_per_step = reward_matrix.groupby('cluster').mean()

# Proceed to plot without dropping 'cluster'
plt.figure(figsize=(12, 6))
sns.heatmap(avg_rewards_per_step, annot=True, cmap='viridis')
plt.title('Average Rewards per Step per Cluster')
plt.xlabel('Step')
plt.ylabel('Cluster')
plt.show()

# Save the DataFrame with analyses
df.to_csv('generated_trajectories_with_rewards_analysis.csv', index=False)
print("DataFrame with analyses saved to 'generated_trajectories_with_rewards_analysis.csv'")
