import pandas as pd
from sklearn.utils import shuffle

# Load the dataset
file_path = '../data/base/cross_validation/train_CV0_size10000.csv'
data = pd.read_csv(file_path, sep='\t')  # Adjust the separator if needed

# Shuffle the data to ensure randomness
data = shuffle(data, random_state=42).reset_index(drop=True)

# Split the data into five roughly equal parts (folds)
num_folds = 5
fold_size = len(data) // num_folds
folds = [data.iloc[i * fold_size: (i + 1) * fold_size] for i in range(num_folds)]

# Handle any remainder by adding the remaining rows to the last fold
if len(data) % num_folds != 0:
    folds[-1] = pd.concat([folds[-1], data.iloc[num_folds * fold_size:]])

# Generate training and testing sets for each fold
for i in range(num_folds):
    # Use the current fold as the testing set
    test_set = folds[i]
    
    # Combine the other folds as the training set
    train_set = pd.concat([folds[j] for j in range(num_folds) if j != i], axis=0)
    
    # Save the training and testing sets
    train_set.to_csv(f'../data/base/cross_validation/train_fold_{i}.csv', index=False, sep='\t')
    test_set.to_csv(f'../data/base/cross_validation/test_fold_{i}.csv', index=False, sep='\t')
    
    # Display basic information about the training and testing sets
    print(f"Fold {i}: Training set = {len(train_set)} rows, Testing set = {len(test_set)} rows")
