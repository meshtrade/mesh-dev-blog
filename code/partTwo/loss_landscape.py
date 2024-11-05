import torch
import torch.nn as nn
import numpy as np
import plotly.graph_objects as go
from typing import List, Tuple

# Define the neural network
class XORNet(nn.Module):
    def __init__(self):
        super(XORNet, self).__init__()
        self.layer1 = nn.Linear(2, 2)
        self.layer2 = nn.Linear(2, 1)
        self.activation = nn.Mish()
        self.sigmoid = nn.Mish()
    
    def forward(self, x):
        x = self.activation(self.layer1(x))
        x = self.sigmoid(self.layer2(x))
        return x

def get_xor_data() -> Tuple[torch.Tensor, torch.Tensor]:
    # XOR input and output
    X = torch.tensor([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=torch.float32)
    y = torch.tensor([[0], [1], [1], [0]], dtype=torch.float32)
    return X, y

def train_model(model: nn.Module, X: torch.Tensor, y: torch.Tensor, epochs: int = 1000000) -> nn.Module:
    criterion = nn.MSELoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.0001)
    
    for epoch in range(epochs):
        optimizer.zero_grad()
        outputs = model(X)
        loss = criterion(outputs, y)
        loss.backward()
        optimizer.step()
        
        if (epoch + 1) % 500 == 0:
            print(f'Epoch {epoch+1}/{epochs}, Loss: {loss.item():.4f}')
    
    return model

def get_random_direction(model: nn.Module) -> List[torch.Tensor]:
    # Generate random direction with the same shape as model parameters
    direction = []
    for param in model.parameters():
        direction.append(torch.randn_like(param))
    return direction

def normalize_direction(direction: List[torch.Tensor], model: nn.Module) -> List[torch.Tensor]:
    # Filter-wise normalization as described in the paper
    normalized_direction = []
    for d, param in zip(direction, model.parameters()):
        norm_d = torch.norm(d)
        norm_param = torch.norm(param)
        normalized_direction.append(d * (norm_param / norm_d))
    return normalized_direction

def compute_loss_grid(
    model: nn.Module,
    direction1: List[torch.Tensor],
    direction2: List[torch.Tensor],
    X: torch.Tensor,
    y: torch.Tensor,
    alpha_range: np.ndarray,
    beta_range: np.ndarray
) -> np.ndarray:
    criterion = nn.MSELoss()
    loss_grid = np.zeros((len(alpha_range), len(beta_range)))
    
    # Store original parameters
    original_params = [param.clone() for param in model.parameters()]

    for i, alpha in enumerate(alpha_range):
        for j, beta in enumerate(beta_range):
            # Restore original parameters
            for param, orig_param in zip(model.parameters(), original_params):
                param.data.copy_(orig_param.data)
            
            # Add perturbation
            for param, d1, d2 in zip(model.parameters(), direction1, direction2):
                param.data.add_(alpha * d1 + beta * d2)
            
            # Compute loss
            with torch.no_grad():
                outputs = model(X)
                loss = criterion(outputs, y)
                loss_grid[i, j] = loss.item()
    
    # Restore original parameters
    for param, orig_param in zip(model.parameters(), original_params):
        param.data.copy_(orig_param.data)
    
    return loss_grid

def visualize_loss_landscape(model: nn.Module, X: torch.Tensor, y: torch.Tensor):
    # Generate and normalize random directions
    d1 = normalize_direction(get_random_direction(model), model)
    d2 = normalize_direction(get_random_direction(model), model)
    
    # Create grid
    alpha_range = np.linspace(-1, 1, 500)
    beta_range = np.linspace(-1, 1, 500)
    
    # Compute loss values
    loss_grid = compute_loss_grid(model, d1, d2, X, y, alpha_range, beta_range)
    
    # Create meshgrid for plotting
    alpha, beta = np.meshgrid(alpha_range, beta_range)
    
    # Create 3D surface plot
    fig = go.Figure(data=[go.Surface(
        x=alpha,
        y=beta,
        z=loss_grid,
        colorscale='viridis'
    )])
    
    fig.update_layout(
        title='Loss Landscape Visualization',
        scene=dict(
            xaxis=dict(title='α'),
            yaxis=dict(title='β'),
            zaxis_title='Loss'
        ),
    )
    
    return fig

def main():
    # Set random seed for reproducibility
    torch.manual_seed(42)
    
    # Get data
    X, y = get_xor_data()
    
    # Create and train model
    model = XORNet()
    model = train_model(model, X, y)
    
    # Test model
    with torch.no_grad():
        test_output = model(X)
        print("\nModel predictions:")
        for input_val, target, pred in zip(X, y, test_output):
            print(f"Input: {input_val.numpy()}, Target: {target.item():.0f}, Prediction: {pred.item():.4f}")
    
    # Visualize loss landscape
    fig = visualize_loss_landscape(model, X, y)
    
    # Save the plot as HTML
    fig.write_html("xor_loss_landscape_loss.html")
    print("\nVisualization saved as 'xor_loss_landscape_loss.html'")

if __name__ == "__main__":
    main()
