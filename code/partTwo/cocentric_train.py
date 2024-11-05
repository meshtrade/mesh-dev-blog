import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.datasets import make_circles
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

# Generate the concentric circles dataset
n_samples = 1000
X, y = make_circles(n_samples=n_samples, noise=0.07, factor=0.5, random_state=42)

# Split into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize the data for better performance
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Convert data to PyTorch tensors
X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train, dtype=torch.float32).unsqueeze(1)  # Reshape for binary output
X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
y_test_tensor = torch.tensor(y_test, dtype=torch.float32).unsqueeze(1)

# Define a simple neural network with one hidden layer of three neurons and a single output
class SimpleNN(nn.Module):
    def __init__(self):
        super(SimpleNN, self).__init__()
        self.fc1 = nn.Linear(2, 3)       # Input layer to hidden layer (2 -> 3)
        self.fc2 = nn.Linear(3, 1)       # Hidden layer to output layer (3 -> 1)
        self.activation = nn.Tanh()      # Activation for hidden layer
        self.output_activation = nn.Sigmoid()  # Sigmoid activation for binary output

    def forward(self, x):
        x = self.activation(self.fc1(x))
        x = self.output_activation(self.fc2(x))  # Sigmoid output for probability
        return x

# Instantiate the model, define loss and optimizer
model = SimpleNN()
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.1)

# Training the model
num_epochs = 2000
for epoch in range(num_epochs):
    # Forward pass
    outputs = model(X_train_tensor)
    loss = criterion(outputs, y_train_tensor)

    # Backward pass and optimization
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    # Print loss for every 20 epochs
    if (epoch+1) % 20 == 0:
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')

# Testing the model
model.eval()  # Set the model to evaluation mode
with torch.no_grad():
    test_outputs = model(X_test_tensor)
    predicted = (test_outputs >= 0.5).float()  # Apply threshold to get binary predictions
    accuracy = (predicted == y_test_tensor).float().mean()
    print(f'Accuracy on test set: {accuracy:.4f}')

# Plotting the decision boundary
def plot_decision_boundary(model, X, y):
    x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
    y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5
    xx, yy = torch.meshgrid(torch.arange(x_min, x_max, 0.01), torch.arange(y_min, y_max, 0.01))
    grid = torch.cat([xx.reshape(-1, 1), yy.reshape(-1, 1)], dim=1).float()
    with torch.no_grad():
        Z = model(grid).reshape(xx.shape)
    Z = Z >= 0.5  # Apply threshold for binary classification

    plt.contourf(xx, yy, Z, alpha=0.3, cmap='coolwarm')
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap='coolwarm', s=20, edgecolor='k')
    plt.title("Decision Boundary")
    plt.show()

# Visualize the decision boundary on the training data
# plot_decision_boundary(model, X_train, y_train)

for name, param in model.named_parameters():
    print(name, " ", param)

