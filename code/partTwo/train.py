import torch
import torch.nn as nn
import torch.optim as optim
import plotly.graph_objects as go

# Define XOR inputs and labels
inputs = torch.tensor([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=torch.float32)
labels = torch.tensor([[0], [1], [1], [0]], dtype=torch.float32)

# Define a neural network with a single hidden layer of two neurons
class XORNet(nn.Module):
    def __init__(self):
        super(XORNet, self).__init__()
        self.layer1 = nn.Linear(2, 2)
        self.layer2 = nn.Linear(2, 1)
        self.activation = nn.Mish() # Adjust activation function here
        self.sigmoid = nn.Mish() # Adjust activation function here
    
    def forward(self, x):
        x = self.activation(self.layer1(x))
        x = self.sigmoid(self.layer2(x))
        return x

# Initialize the model, loss function, and optimizer
model = XORNet()
criterion = nn.MSELoss()  
optimizer = optim.SGD(model.parameters(), lr=0.1) # Learning Rate

# Training loop
epochs = 40000
loss_values = []  # To store loss for each epoch

for epoch in range(epochs):
    optimizer.zero_grad()  # Clear the gradients
    outputs = model(inputs)  # Forward pass
    loss = criterion(outputs, labels)  # Compute the loss
    loss.backward()  # Backpropagation
    optimizer.step()  # Update the weights
    
    loss_values.append(loss.item())  # Store the loss value
    
    # Print the loss every 1000 epochs
    if (epoch + 1) % 100 == 0:
        print(f"Epoch [{epoch + 1}/{epochs}], Loss: {loss.item():.4f}")

# Plot the loss values over epochs
fig = go.Figure()
fig.add_trace(go.Scatter(x=list(range(epochs)), y=loss_values, mode='lines', name='Loss'))
fig.update_layout(
    title="Loss Over Epochs",
    xaxis_title="Epoch",
    yaxis_title="Loss",
)
fig.write_html("loss_run.html")

# Test the model
with torch.no_grad():
    predictions = model(inputs)  
    print("Predictions:")
    print(predictions)
    print("Labels:")
    print(labels)

for name, param in model.named_parameters():
    print(name, " ", param)
