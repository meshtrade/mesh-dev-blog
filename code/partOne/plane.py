import numpy as np
import plotly.graph_objects as go

# Input data for AND gate (extended with bias as the third dimension)
X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
y = np.array([0, 0, 0, 1])  # AND gate outputs

# Assume learned weights
w1, w2 = 0.15, 0.4  # Weights

# Define a plane equation
x1_vals = np.linspace(0, 1, 10)
print(x1_vals)
x2_vals = np.linspace(0, 1, 10)
x1_vals, x2_vals = np.meshgrid(x1_vals, x2_vals)
x3_vals = -(w1 * x1_vals + w2 * x2_vals) + 0.5

# Create a 3D scatter plot for the input points
scatter = go.Scatter3d(
    x=X[:, 0], y=X[:, 1], z=y,
    mode='markers',
    marker=dict(size=8, color=y, colorscale='Viridis'),
    name='Input points'
)

# Create a 3D surface plot for the decision boundary (plane)
surface = go.Surface(
    x=x1_vals, y=x2_vals, z=x3_vals,
    opacity=0.5, colorscale='Viridis',
    name='Separating Plane'
)

# Set up the layout and axis labels
layout = go.Layout(
    scene=dict(
        xaxis=dict(title='x1'),
        yaxis=dict(title='x2'),
        zaxis=dict(title='Output'),
    ),
    title='AND Gate Perceptron Decision Boundary'
)

# Create the figure and display it
fig = go.Figure(data=[scatter, surface], layout=layout)

fig.write_html("plane.html")
