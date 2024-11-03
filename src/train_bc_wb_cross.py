import time
import torch
import numpy as np
import csv
import pandas as pd
from utils.load_data import load_path_feature, load_link_feature, minmax_normalization, load_test_traj
from utils.evaluation import evaluate_model, evaluate_log_prob
from network_env import RoadWorld
from model.policy import PolicyCNN
import wandb

if __name__ == '__main__':
    # Initialize Weights & Biases
    wandb.init(project="RCM-BC", entity="reneelin2024", resume="allow")
    config = wandb.config
    config.learning_rate = 1e-3
    config.batch_size = 32
    config.max_iter_num = 300
    config.log_interval = 10

    # Set hyperparameters
    learning_rate = config.learning_rate
    batch_size = config.batch_size
    max_iter_num = config.max_iter_num
    log_interval = config.log_interval

    max_length = 50  # maximum path length for recursively generating next link until destination is reached
    size = 10000  # size of training data [100, 1000, 10000]
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    edge_p = "../data/base/edge.txt"
    network_p = "../data/base/transit.npy"
    path_feature_p = "../data/base/feature_od.npy"
    
    # Initialize road environment
    env = RoadWorld(network_p, edge_p)
    
    # Load path-level and link-level features
    path_feature, path_max, path_min = load_path_feature(path_feature_p)
    edge_feature, edge_max, edge_min = load_link_feature(edge_p)
    path_feature = minmax_normalization(path_feature, path_max, path_min)
    edge_feature = minmax_normalization(edge_feature, edge_max, edge_min)
    path_feature_pad = np.zeros((env.n_states, env.n_states, path_feature.shape[2]))
    path_feature_pad[:path_feature.shape[0], :path_feature.shape[1], :] = path_feature
    edge_feature_pad = np.zeros((env.n_states, edge_feature.shape[1]))
    edge_feature_pad[:edge_feature.shape[0], :] = edge_feature

    # Iterate over each fold
    num_folds = 5
    metrics_list = []

    for cv in range(num_folds):
        # Define paths for current fold
        train_p = f"../data/base/cross_validation/train_fold_{cv}.csv"
        test_p = f"../data/base/cross_validation/test_fold_{cv}.csv"
        model_p = f"../trained_models/base/bc_fold_{cv}.pt"

        # Initialize model
        CNNMODEL = PolicyCNN(
            env.n_actions,
            env.policy_mask,
            env.state_action,
            path_feature_pad,
            edge_feature_pad,
            path_feature_pad.shape[-1] + edge_feature_pad.shape[-1] + 1,
            env.pad_idx
        ).to(device)
        CNNMODEL.to_device(device)
        criterion = torch.nn.NLLLoss()
        optimizer = torch.optim.Adam(CNNMODEL.parameters(), lr=learning_rate)

        # Load training data for the current fold
        train_trajs = env.import_demonstrations_step(train_p)
        train_state_list, train_des_list, train_action_list = [], [], []
        for episode in train_trajs:
            for x in episode:
                train_state_list.append(x.cur_state)
                train_des_list.append(episode[-1].next_state)
                train_action_list.append(x.action)
        x_state_train = torch.LongTensor(train_state_list).to(device)
        x_des_train = torch.LongTensor(train_des_list).to(device)
        y_train = torch.LongTensor(train_action_list).to(device)

        # Start training for the current fold and log to W&B and CSV
        start_time = time.time()
        with open(f'bc_training_log_fold_{cv}.csv', 'w', newline='') as csvfile:
            fieldnames = [
                'Iteration', 'Elapsed Time', 'Training Loss',
                'Training Edit Distance', 'Testing Edit Distance',
                'Training BLEU Score', 'Testing BLEU Score',
                'Training JS Distance', 'Testing JS Distance'
            ]
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()

            for it in range(1, max_iter_num + 1):
                epoch_loss = 0
                num_batches = 0
                for i in range(0, x_state_train.shape[0], batch_size):
                    batch_right = min(i + batch_size, x_state_train.shape[0])
                    sampled_x_state_train = x_state_train[i:batch_right]
                    sampled_x_des_train = x_des_train[i:batch_right]
                    sampled_y_train = y_train[i:batch_right].contiguous().view(-1)
                    y_est = CNNMODEL.get_action_log_prob(sampled_x_state_train, sampled_x_des_train)
                    loss = criterion(y_est.view(-1, y_est.size(1)), sampled_y_train)
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                    epoch_loss += loss.item()
                    num_batches += 1
                
                avg_loss = epoch_loss / num_batches
                elapsed_time = time.time() - start_time

                # Evaluate the model on both training and testing sets at the log interval
                if it % log_interval == 0 or it == 1:
                    # Evaluate on the training set
                    train_traj, train_od = load_test_traj(train_p)
                    train_od = torch.from_numpy(train_od).long().to(device)
                    train_edit_dist, train_bleu_score, train_js_distance = evaluate_model(
                        train_od, train_traj, CNNMODEL, env
                    )
                    
                    # Log training metrics to console and W&B
                    print(
                        f"Fold {cv} | Iteration {it}/{max_iter_num} | Training Metrics | "
                        f"Edit Distance: {train_edit_dist:.4f} | BLEU Score: {train_bleu_score:.4f} | "
                        f"JS Distance: {train_js_distance:.4f}"
                    )
                    
                    wandb.log({
                        'Fold': cv,
                        'Iteration': it,
                        'Training Loss': avg_loss,
                        'Elapsed Time': elapsed_time,
                        'Training Edit Distance': train_edit_dist,
                        'Training BLEU Score': train_bleu_score,
                        'Training JS Distance': train_js_distance
                    })
                    
                    # Evaluate on the testing set
                    target_traj, target_od = load_test_traj(test_p)
                    target_od = torch.from_numpy(target_od).long().to(device)
                    edit_dist, bleu_score, js_distance = evaluate_model(
                        target_od, target_traj, CNNMODEL, env
                    )
                    
                    # Log testing metrics to console and W&B
                    print(
                        f"Fold {cv} | Iteration {it}/{max_iter_num} | Testing Metrics | "
                        f"Edit Distance: {edit_dist:.4f} | BLEU Score: {bleu_score:.4f} | "
                        f"JS Distance: {js_distance:.4f}"
                    )

                    wandb.log({
                        'Fold': cv,
                        'Iteration': it,
                        'Testing Edit Distance': edit_dist,
                        'Testing BLEU Score': bleu_score,
                        'Testing JS Distance': js_distance
                    })

                    # Write to CSV file
                    writer.writerow({
                        'Iteration': it,
                        'Elapsed Time': elapsed_time,
                        'Training Loss': avg_loss,
                        'Training Edit Distance': train_edit_dist,
                        'Training BLEU Score': train_bleu_score,
                        'Training JS Distance': train_js_distance,
                        'Testing Edit Distance': edit_dist,
                        'Testing BLEU Score': bleu_score,
                        'Testing JS Distance': js_distance
                    })

            # Save the model and log it to W&B
            torch.save(CNNMODEL.state_dict(), model_p)
            try:
                artifact = wandb.Artifact(f"trained_model_fold_{cv}", type="model")
                artifact.add_file(model_p)
                wandb.log_artifact(artifact)
            except Exception as e:
                print(f"Warning: Failed to log model artifact in W&B due to: {e}")

        # Final evaluation for the current fold
        CNNMODEL.load_state_dict(torch.load(model_p, map_location=torch.device('cpu')))
        print(f'Total training time for fold {cv}:', time.time() - start_time)
        target_traj, target_od = load_test_traj(test_p)
        target_od = torch.from_numpy(target_od).long().to(device)
        eval_metrics = evaluate_model(target_od, target_traj, CNNMODEL, env)
        metrics_list.append(eval_metrics)
        print(f'Total evaluation time for fold {cv}:', time.time() - start_time)

    # Calculate average metrics across all folds
    avg_edit_dist = np.mean([metrics[0] for metrics in metrics_list])
    avg_bleu_score = np.mean([metrics[1] for metrics in metrics_list])
    avg_js_distance = np.mean([metrics[2] for metrics in metrics_list])
    print(f"Average Edit Distance: {avg_edit_dist:.4f}")
    print(f"Average BLEU Score: {avg_bleu_score:.4f}")
    print(f"Average JS Distance: {avg_js_distance:.4f}")
