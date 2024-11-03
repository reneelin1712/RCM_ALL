import os
import numpy as np
import torch
import matplotlib.pyplot as plt
from network_env import RoadWorld
from utils.load_data import load_test_traj, ini_od_dist, load_path_feature, load_link_feature, minmax_normalization
from model.policy import PolicyCNN
from captum.attr import (
    IntegratedGradients,
    Saliency,
    InputXGradient,
    DeepLift,
    GradientShap,
    GuidedBackprop,
    Deconvolution,
    Occlusion,
    FeatureAblation,
    # FeaturePermutation,  # Removed due to batch size constraints
    ShapleyValueSampling,
    # Lime,
    # KernelShap
)
from tqdm import tqdm  # Added for progress bar

def load_bc_model(model_path, device, env, path_feature_pad, edge_feature_pad):
    # Initialize the PolicyCNN model
    policy_net = PolicyCNN(
        env.n_actions,
        env.policy_mask,
        env.state_action,
        path_feature_pad,
        edge_feature_pad,
        path_feature_pad.shape[-1] + edge_feature_pad.shape[-1] + 1,
        env.pad_idx
    ).to(device)
    
    # Load the model state dictionary
    policy_net.load_state_dict(torch.load(model_path, map_location=device))
    policy_net.eval()  # Set to evaluation mode
    policy_net.to_device(device)
    return policy_net

def get_cnn_input(policy_net, state, des, device):
    state = torch.tensor([state], dtype=torch.long).to(device)
    des = torch.tensor([des], dtype=torch.long).to(device)
    # Process features to get the CNN input
    input_data = policy_net.process_features(state, des)
    return input_data

# Define a wrapper module for the policy network
class PolicyWrapper(torch.nn.Module):
    def __init__(self, policy_net, state_tensor):
        super(PolicyWrapper, self).__init__()
        self.policy_net = policy_net
        self.state_tensor = state_tensor

    def forward(self, input_data):
        x = self.policy_net.forward(input_data)
        x_mask = self.policy_net.policy_mask[self.state_tensor]
        x = x.masked_fill((1 - x_mask).bool(), -1e32)
        action_probs = torch.softmax(x, dim=1)
        return action_probs

def interpret_bc_model():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Ensure output directories exist
    output_dir = 'output_img/bc_attribution'
    os.makedirs(output_dir, exist_ok=True)

    # Path settings (adjust paths and variables as necessary)
    cv = 0  # Cross-validation index
    size = 10000  # Size of the training data
    model_path = f"../trained_models/base/bc_CV{cv}_size{size}.pt"
    edge_p = "../data/base/edge.txt"
    network_p = "../data/base/transit.npy"
    path_feature_p = "../data/base/feature_od.npy"
    test_p = f"../data/base/cross_validation/test_CV{cv}.csv"

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

    # Load the BC model
    policy_net = load_bc_model(model_path, device, env, path_feature_pad, edge_feature_pad)

    # Load test trajectories
    test_trajs, test_od = load_test_traj(test_p)

    # Define attribution methods
    attribution_methods = [
        ('IntegratedGradients', IntegratedGradients),
        ('Saliency', Saliency),
        ('InputXGradient', InputXGradient),
        ('DeepLift', DeepLift),
        # ('DeepLiftShap', DeepLiftShap),  # Requires multiple inputs
        ('GradientShap', GradientShap),
        ('GuidedBackprop', GuidedBackprop),
        ('Deconvolution', Deconvolution),
        ('Occlusion', Occlusion),
        ('FeatureAblation', FeatureAblation),
        # ('FeaturePermutation', FeaturePermutation),  # Removed due to batch size constraints
        ('ShapleyValueSampling', ShapleyValueSampling),
        # ('Lime', Lime),  # Lime and KernelShap require additional setup
        # ('KernelShap', KernelShap)
    ]

    # Initialize total attributions dictionary
    total_attributions = {method_name: None for method_name, _ in attribution_methods}
    count = 0

    # Added progress bar over trajectories
    for traj_idx, traj in enumerate(tqdm(test_trajs, desc='Trajectories')):
        states_list = [int(s) for s in traj[:-1]]
        destination = int(traj[-1])

        # Added progress bar over states
        for state_idx, state in enumerate(tqdm(states_list, desc='States', leave=False)):
            input_data = get_cnn_input(policy_net, state, destination, device)
            input_data.requires_grad = True

            state_tensor = torch.tensor([state], dtype=torch.long).to(device)
            # Create wrapper model
            wrapped_model = PolicyWrapper(policy_net, state_tensor)

            with torch.no_grad():
                output = wrapped_model(input_data)
            predicted_action = torch.argmax(output, dim=1)

            # Loop over attribution methods
            for method_name, method_class in attribution_methods:
                method = method_class(wrapped_model)
                # Handle methods requiring baselines or additional inputs
                if method_name in ['DeepLift', 'GradientShap']:
                    baseline = torch.zeros_like(input_data)
                    if method_name == 'GradientShap':
                        # For GradientShap, we can provide multiple baselines
                        baselines = torch.cat([input_data * 0, input_data * 1], dim=0)
                        attributions = method.attribute(input_data, baselines=baselines, target=predicted_action)
                    else:
                        attributions = method.attribute(input_data, baselines=baseline, target=predicted_action)
                elif method_name == 'Occlusion':
                    # Adjust sliding window shapes to fit input dimensions
                    sliding_window_shapes = tuple(min(s, i) for s, i in zip((1, 1, 1), input_data.shape))
                    attributions = method.attribute(input_data, target=predicted_action, sliding_window_shapes=sliding_window_shapes)
                else:
                    attributions = method.attribute(input_data, target=predicted_action)
                
                # Convert to numpy array
                attr_np = attributions.squeeze().cpu().detach().numpy()
                
                # Accumulate attributions
                if total_attributions[method_name] is None:
                    total_attributions[method_name] = attr_np
                else:
                    total_attributions[method_name] += attr_np

            count += 1

    # Compute average attributions
    avg_attributions = {method_name: total_attr / count for method_name, total_attr in total_attributions.items()}

    # Sum over the channels to get a single 2D map
    for method_name, avg_attr in avg_attributions.items():
        avg_attr_sum = np.sum(avg_attr, axis=0)

        # Save averaged attribution image
        plt.figure(figsize=(6, 4))
        plt.imshow(avg_attr_sum, cmap='hot')
        plt.title(f'Averaged {method_name} Attribution (BC Policy)')
        plt.colorbar()
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f'averaged_{method_name.lower()}_attribution_bc.png'))
        plt.close()

        num_channels = avg_attr.shape[0]
        for channel_idx in range(num_channels):
            channel_attr = avg_attr[channel_idx]

            # Save channel-wise attribution
            plt.figure(figsize=(6, 4))
            plt.imshow(channel_attr, cmap='hot')
            plt.title(f'Averaged {method_name} Attribution - Feature {channel_idx} (BC Policy)')
            plt.colorbar()
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, f'averaged_{method_name.lower()}_attribution_feature_{channel_idx}_bc.png'))
            plt.close()

    print("Interpretation complete. Results saved in the 'output_img/bc_attribution' directory.")

if __name__ == '__main__':
    interpret_bc_model()
