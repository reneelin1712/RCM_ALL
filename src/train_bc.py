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
    learning_rate = 1e-3
    n_iters = 500  # number of training iterations
    batch_size = 32  # training batch size
    max_length = 50  # maximum path length for recursively generating next link until destination is reached
    cv = 0  # cross validation process [0, 1, 2, 3, 4]
    size = 10000  # size of training data [100, 1000, 10000]
    max_iter_num = 500
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    log_interval = 10  # interval for logging

    edge_p = "../data/base/edge.txt"
    network_p = "../data/base/transit.npy"
    path_feature_p = "../data/base/feature_od.npy"
    train_p = "../data/base/cross_validation/train_CV%d_size%d.csv" % (cv, size)
    test_p = "../data/base/cross_validation/test_CV%d.csv" % cv
    model_p = "../trained_models/base/bc_CV%d_size%d.pt" % (cv, size)
    """initialize road environment"""
    env = RoadWorld(network_p, edge_p)
    """load path-level and link-level feature"""
    path_feature, path_max, path_min = load_path_feature(path_feature_p)
    edge_feature, edge_max, edge_min = load_link_feature(edge_p)
    path_feature = minmax_normalization(path_feature, path_max, path_min)
    path_feature_pad = np.zeros((env.n_states, env.n_states, path_feature.shape[2]))
    path_feature_pad[:path_feature.shape[0], :path_feature.shape[1], :] = path_feature
    edge_feature = minmax_normalization(edge_feature, edge_max, edge_min)
    edge_feature_pad = np.zeros((env.n_states, edge_feature.shape[1]))
    edge_feature_pad[:edge_feature.shape[0], :] = edge_feature
    """initiate model"""
    CNNMODEL = PolicyCNN(env.n_actions, env.policy_mask, env.state_action, path_feature_pad, edge_feature_pad,
                         path_feature_pad.shape[-1] + edge_feature_pad.shape[-1] + 1, env.pad_idx).to(device)
    CNNMODEL.to_device(device)
    criterion = torch.nn.NLLLoss()
    optimizer = torch.optim.Adam(CNNMODEL.parameters(), lr=learning_rate)
    """load train data"""
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
    """train model"""
    start_time = time.time()
    
    # Open a CSV file for logging
    with open('bc_training_log.csv', 'w', newline='') as csvfile:
        fieldnames = ['Iteration', 'Elapsed Time', 'Training Loss']
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
            if it % log_interval == 0 or it == 1:
                elapsed_time = time.time() - start_time
                print(f"Iteration {it}/{max_iter_num} | Elapsed Time: {elapsed_time:.2f}s | Loss: {avg_loss:.4f}")
                
                # Optionally, you can evaluate the model here and include metrics
                # For example:
                # target_traj, target_od = load_test_traj(test_p)
                # target_od = torch.from_numpy(target_od).long().to(device)
                # evaluation_metric = evaluate_model(target_od, target_traj, CNNMODEL, env)
                # For now, we'll skip this step to save time during training
                
                # Write to CSV
                writer.writerow({
                    'Iteration': it,
                    'Elapsed Time': elapsed_time,
                    'Training Loss': avg_loss,
                })
                
    torch.save(CNNMODEL.state_dict(), model_p)
    """evaluate model"""
    CNNMODEL.load_state_dict(torch.load(model_p, map_location=torch.device('cpu')))
    print('Total training time:', time.time() - start_time)
    target_traj, target_od = load_test_traj(test_p)
    target_od = torch.from_numpy(target_od).long().to(device)
    evaluate_model(target_od, target_traj, CNNMODEL, env)
    print('Total evaluation time:', time.time() - start_time)
    """evaluate log prob"""
    test_trajs = env.import_demonstrations_step(test_p)
    evaluate_log_prob(test_trajs, CNNMODEL)
