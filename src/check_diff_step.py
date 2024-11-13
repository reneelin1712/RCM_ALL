import pandas as pd

# Read the Excel File into a DataFrame
df = pd.read_excel('same_OD_different_paths.xlsx')

# Initialize New Columns to Store Results
df['Different_From_Index'] = None
df['Different_From_Node'] = None
df['Same_Again_From_Index'] = None
df['Same_Again_From_Node'] = None

# Group the Data by Origin and Destination
grouped = df.groupby(['Origin', 'Destination'])

# Create a Dictionary to Map OD Pairs to Their Test Trajectories
od_test_trajectories = {}

for (origin, destination), group in grouped:
    # Get unique Test Trajectories for this OD pair
    test_trajectories = group['Test Trajectory'].unique()
    od_test_trajectories[(origin, destination)] = test_trajectories

# Process Each Row to Compare Trajectories
for index, row in df.iterrows():
    learner_trajectory_str = str(row['Learner Trajectory'])
    origin = row['Origin']
    destination = row['Destination']

    # Get all Test Trajectories with the same OD, excluding the current Learner Trajectory
    test_trajectories = od_test_trajectories.get((origin, destination), [])
    test_trajectories = [t for t in test_trajectories if t != learner_trajectory_str]

    if test_trajectories:
        # Use the first different Test Trajectory for comparison
        test_trajectory_str = test_trajectories[0]

        # Split the trajectories into lists of node IDs
        test_trajectory = test_trajectory_str.split('_')
        learner_trajectory = learner_trajectory_str.split('_')

        # Find where they start to differ
        diff_start_index = None
        diff_start_node = None
        min_length = min(len(test_trajectory), len(learner_trajectory))
        for i in range(min_length):
            if test_trajectory[i] != learner_trajectory[i]:
                diff_start_index = i
                diff_start_node = learner_trajectory[i]
                break
        if diff_start_index is None and len(test_trajectory) != len(learner_trajectory):
            diff_start_index = min_length
            diff_start_node = learner_trajectory[min_length] if len(learner_trajectory) > min_length else None

        # Find where they become the same again
        same_again_index = None
        same_again_node = None
        reversed_test = test_trajectory[::-1]
        reversed_learner = learner_trajectory[::-1]
        min_rev_length = min(len(reversed_test), len(reversed_learner))
        for i in range(min_rev_length):
            if reversed_test[i] != reversed_learner[i]:
                same_again_index = len(learner_trajectory) - i
                same_again_node = learner_trajectory[same_again_index] if same_again_index < len(learner_trajectory) else None
                break
        if same_again_index is None and len(reversed_test) != len(reversed_learner):
            same_again_index = len(learner_trajectory) - min_rev_length
            same_again_node = learner_trajectory[same_again_index] if same_again_index < len(learner_trajectory) else None

        # Record the results in the DataFrame
        df.loc[index, 'Different_From_Index'] = diff_start_index
        df.loc[index, 'Different_From_Node'] = diff_start_node
        df.loc[index, 'Same_Again_From_Index'] = same_again_index
        df.loc[index, 'Same_Again_From_Node'] = same_again_node
    else:
        # No different Test Trajectory exists for comparison
        pass

# Save the Updated DataFrame to a New Excel File
df.to_excel('same_OD_different_paths_processed.xlsx', index=False)


# import pandas as pd

# df = pd.read_excel('same_OD_different_paths.xlsx')

# df['Different_From_Index'] = None
# df['Different_From_Node'] = None
# df['Same_Again_From_Index'] = None
# df['Same_Again_From_Node'] = None

# for index, row in df.iterrows():
#     test_trajectory_str = str(row['Test Trajectory'])
#     learner_trajectory_str = str(row['Learner Trajectory'])
#     origin = row['Origin']
#     destination = row['Destination']

#     # Only process if trajectories are different and have the same OD
#     if test_trajectory_str != learner_trajectory_str and row['Origin'] == row['Origin'] and row['Destination'] == row['Destination']:
#         # Split the trajectories into lists of node IDs
#         test_trajectory = test_trajectory_str.split('_')
#         learner_trajectory = learner_trajectory_str.split('_')

#         # Find where they start to differ
#         diff_start_index = None
#         diff_start_node = None
#         min_length = min(len(test_trajectory), len(learner_trajectory))
#         for i in range(min_length):
#             if test_trajectory[i] != learner_trajectory[i]:
#                 diff_start_index = i
#                 diff_start_node = learner_trajectory[i]
#                 break
#         if diff_start_index is None and len(test_trajectory) != len(learner_trajectory):
#             diff_start_index = min_length
#             diff_start_node = learner_trajectory[min_length] if len(learner_trajectory) > min_length else None

#         # Find where they become the same again
#         same_again_index = None
#         same_again_node = None
#         reversed_test = test_trajectory[::-1]
#         reversed_learner = learner_trajectory[::-1]
#         min_rev_length = min(len(reversed_test), len(reversed_learner))
#         for i in range(min_rev_length):
#             if reversed_test[i] != reversed_learner[i]:
#                 same_again_index = len(learner_trajectory) - i
#                 same_again_node = learner_trajectory[same_again_index] if same_again_index < len(learner_trajectory) else None
#                 break
#         if same_again_index is None and len(reversed_test) != len(reversed_learner):
#             same_again_index = len(learner_trajectory) - min_rev_length
#             same_again_node = learner_trajectory[same_again_index] if same_again_index < len(learner_trajectory) else None

#         # Record the results in the DataFrame
#         df.loc[index, 'Different_From_Index'] = diff_start_index
#         df.loc[index, 'Different_From_Node'] = diff_start_node
#         df.loc[index, 'Same_Again_From_Index'] = same_again_index
#         df.loc[index, 'Same_Again_From_Node'] = same_again_node
#     else:
#         # Paths are the same or OD is different; no action needed
#         pass

# df.to_excel('same_OD_different_paths_processed.xlsx', index=False)
