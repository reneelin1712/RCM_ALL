import torch
import torch.nn.functional as F
import numpy as np
import pandas as pd
import shap
import matplotlib.pyplot as plt
import os

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
    policy_net.load_state_dict(model_dict["Policy"])
    value_net.load_state_dict(model_dict["Value"])
    discriminator_net.load_state_dict(model_dict["Discrim"])

    policy_net.eval()
    value_net.eval()
    discriminator_net.eval()

    return policy_net, value_net, discriminator_net


def get_cnn_input(policy_net, state, des, device):
    state = torch.tensor([state], dtype=torch.long).to(device)
    des = torch.tensor([des], dtype=torch.long).to(device)
    # Process features to get the CNN input
    input_data = policy_net.process_features(state, des)
    return input_data


def interpret_model():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Feature names
    feature_names = [
        "Number of links",
        "Total length",
        "Number of left turns",
        "Number of right turns",
        "Number of U-turns",
        "Number of residential roads",
        "Number of primary roads",
        "Number of unclassified roads",
        "Number of tertiary roads",
        "Number of living_street roads",
        "Number of secondary roads",
        "Mask feature",
        "Edge length",
        "Highway type: residential",
        "Highway type: primary",
        "Highway type: unclassified",
        "Highway type: tertiary",
        "Highway type: living_street",
        "Highway type: secondary",
        "Neighbor mask",
    ]

    # Paths to data and models
    data_p = "../data/base/cross_validation/train_CV0_size10000.csv"
    model_path = "../trained_models/base/bleu90.pt"
    edge_p = "../data/base/edge.txt"
    network_p = "../data/base/transit.npy"
    path_feature_p = "../data/base/feature_od.npy"
    generated_trajs_csv = "./eva/generated_trajectories_with_rewards.csv"

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

    # Read generated trajectories
    df_generated = pd.read_csv(generated_trajs_csv)

    # Prepare data for SHAP
    input_data_list = []
    state_list = []
    destination_list = []

    for idx, row in df_generated.iterrows():
        trajectory_str = row["generated_trajectory"]
        actions_taken = eval(row["actions_taken"])
        destination_node = int(row["destination"])

        trajectory = [int(s) for s in trajectory_str.strip().split("_")]
        states = trajectory[:-1]

        for state in states:
            input_data = get_cnn_input(policy_net, state, destination_node, device)
            input_data_np = input_data.cpu().detach().numpy().squeeze(0)
            input_data_list.append(input_data_np)
            state_list.append(state)
            destination_list.append(destination_node)

    input_array = np.array(input_data_list)  # Shape: [num_samples, channels, height, width]
    input_array_flat = input_array.reshape(len(input_data_list), -1)  # Flatten input data

    # Expand feature names to match flattened input
    expanded_feature_names = [
        f"{feature}_{h}_{w}"
        for feature in feature_names
        for h in range(input_array.shape[2])
        for w in range(input_array.shape[3])
    ]

    # Ensure expanded_feature_names aligns with the number of flattened features
    assert len(expanded_feature_names) == input_array_flat.shape[1]

    # SHAP for policy network
    policy_background = shap.kmeans(input_array_flat, 50)

    def policy_predict(input_array_flat):
        outputs = []
        for i in range(len(input_array_flat)):
            reshaped_input = input_array_flat[i].reshape(input_array.shape[1:])
            input_tensor = torch.tensor(reshaped_input, dtype=torch.float32).unsqueeze(0).to(device)
            with torch.no_grad():
                output = policy_net(input_tensor)
                outputs.append(output.cpu().numpy().flatten())
        return np.array(outputs)

    policy_explainer = shap.KernelExplainer(policy_predict, policy_background)
    policy_shap_values = policy_explainer.shap_values(input_array_flat[:30])

    shap.summary_plot(policy_shap_values[0], input_array_flat[:30], feature_names=expanded_feature_names)


if __name__ == "__main__":
    interpret_model()
