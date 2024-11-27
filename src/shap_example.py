import torch
import torch.nn as nn
import torch.nn.functional as F
import shap
import numpy as np

class SimpleNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(SimpleNN, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, output_size)
        
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# Instantiate the model
input_size = 10
hidden_size = 5
output_size = 1
model = SimpleNN(input_size, hidden_size, output_size)

# Create some dummy data
X = np.random.randn(100, input_size)
y = np.random.randn(100, output_size)

# Convert data to tensors
X_tensor = torch.from_numpy(X).float()
y_tensor = torch.from_numpy(y).float()

# Define loss and optimizer
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.0003)

# Training loop
for epoch in range(100):
    optimizer.zero_grad()
    outputs = model(X_tensor)
    loss = criterion(outputs, y_tensor)
    loss.backward()
    optimizer.step()

# Select a background dataset for SHAP (e.g., a subset of X)
background = X[np.random.choice(X.shape[0], 20, replace=False)]

# Convert background to tensors
background_tensor = torch.from_numpy(background).float()

# Create a SHAP explainer using GradientExplainer
explainer = shap.GradientExplainer(model, background_tensor)

# Explain a few samples
test_samples = X[:10]
test_samples_tensor = torch.from_numpy(test_samples).float()

# Compute SHAP values
shap_values = explainer.shap_values(test_samples_tensor)

# Plot the SHAP values
shap.summary_plot(shap_values, test_samples)
