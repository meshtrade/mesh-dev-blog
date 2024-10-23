import plotly.graph_objs as go
import numpy as np

# Define the XOR points: (input1, input2, output)
points = np.array([[0, 0, 0],  # (0, 0) -> 0
                   [0, 1, 1],  # (0, 1) -> 1
                   [1, 0, 1],  # (1, 0) -> 1
                   [1, 1, 0]]) # (1, 1) -> 0

# Create a scatter plot of the points
scatter = go.Scatter3d(
    x=points[:, 0],
    y=points[:, 1],
    z=points[:, 2],
    mode='markers',
    marker=dict(size=8, color=['blue', 'red', 'red', 'blue']),
    name="XOR Points"
)

# Define a nonlinear surface to separate output=1 points from output=0 points
# We'll use a simple curved surface for visualization purposes
xx, yy = np.meshgrid(np.linspace(-0.5, 1.5, 20), np.linspace(-0.5, 1.5, 20))
zz = np.sin(np.pi * xx * yy)  # Arbitrary separating surface for XOR

# Create a surface plot for the separation
surface = go.Surface(
    x=xx,
    y=yy,
    z=zz,
    colorscale='Viridis',
    opacity=0.6,
    showscale=False,
    name="Separating Surface"
)

# Set up the layout for the plot
layout = go.Layout(
    scene=dict(
        xaxis=dict(title='A'),
        yaxis=dict(title='B'),
        zaxis=dict(title='Q'),
    ),
    title="XOR Gate Input/Output"
)

# Combine the plots
fig = go.Figure(data=[scatter, surface], layout=layout)

# Display the plot
fig.write_html("xor.html")
