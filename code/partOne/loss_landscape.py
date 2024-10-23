import numpy as np
import plotly.graph_objs as go

# AND gate truth table
X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])  # Inputs
y = np.array([0, 0, 0, 1])  # AND gate outputs

def step(x):
    step_x = []
    for d in x:
        step_x.append(1 if d > 0.0 else 0)
    return step_x

def perceptron(X, w):
    # Weighted sum of inputs
    return step(np.dot(X, w) - 0.6)

def mse_loss(y_true, y_pred):
    return np.square(y_true - y_pred).mean()

# Loss landscape over a grid of weight values
def loss_landscape(X, y, w_range):
    W1, W2 = np.meshgrid(w_range, w_range)
    Loss = np.zeros(W1.shape)

    # Calculate loss for each weight combination
    for i in range(W1.shape[0]):
        for j in range(W1.shape[1]):
            w = np.array([W1[i, j], W2[i, j]])
            y_pred = perceptron(X, w)
            loss = mse_loss(y, y_pred)
            Loss[i, j] = loss

    return W1, W2, Loss

# Define the range of weights to explore
w_range = np.linspace(-2, 2, 100)

# Calculate the loss landscape
W1, W2, Loss = loss_landscape(X, y, w_range)

# Create 3D surface plot with Plotly
surface = go.Surface(
    x=W1, 
    y=W2, 
    z=Loss, 
    colorscale='Viridis',
    contours={
        "z": {"show": True, "usecolormap": True, "highlightcolor": "limegreen", "project_z": True}
    }
)

# Points to plot on the loss landscape
points = np.array([
    [-1.25, 1.1], # Epoch = 0      
    [-0.85, 1.1], # Epoch = 4
    [-0.45, 1.1], # Epoch = 8
    [-0.15, 1.0], # Epoch = 12
    [0.05, 0.8], # Epoch = 16
    [0.5, 0.5],
])

# Calculate the corresponding loss for each point
losses = [mse_loss(y, perceptron(X, w)) for w in points]

# Create scatter plot for the points
scatter = go.Scatter3d(
    x=points[:, 0],  # Weight 1 values
    y=points[:, 1],  # Weight 2 values
    z=losses,        # Loss values
    mode='lines+markers',
    marker=dict(size=10, color='black'),
    name='Points'
)

layout = go.Layout(
    title='Loss Landscape of a Perceptron for AND Gate',
    scene=dict(
        xaxis_title='Weight 1',
        yaxis_title='Weight 2',
        zaxis_title='Loss (MSE)'
    )
)

# Create the figure
fig = go.Figure(data=[surface], layout=layout)

# Save the figure as an interactive HTML file
fig.write_html('loss_landscape.html')

