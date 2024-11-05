import plotly.graph_objs as go
import numpy as np

# Define the points representing XOR gate inputs and outputs
points = np.array([[0, 0, 0], [0, 1, 1], [1, 0, 1], [1, 1, 0]])

# Create a scatter plot of the points
scatter = go.Scatter3d(
    x=points[:, 0],
    y=points[:, 1],
    z=points[:, 2],
    mode='markers',
    marker=dict(size=8, color=[0,0,0,1], colorscale='Viridis'),
    name="XOR Gate IO",
)

# Define the plane equation (the plane will be x + y + z = 1)
xx, yy = np.meshgrid(np.linspace(0.0, 1, 3), np.linspace(0.0, 1, 3))
zz = np.full(xx.shape, 0.5)

# Create a surface plot for the plane
plane = go.Surface(
    x=xx,
    y=yy,
    z=zz,
    colorscale='Viridis',
    opacity=0.5,
    name="Separating Plane"
)

# Set up the layout for the plot
layout = go.Layout(
    scene=dict(
        xaxis=dict(title='X-axis'),
        yaxis=dict(title='Y-axis'),
        zaxis=dict(title='Z-axis'),
    ),
    title="XOR Gate Input/Output Plot"
)

# Combine the plots
fig = go.Figure(data=[scatter, plane], layout=layout)

# Save the figure as an HTML file
fig.write_html("xorgateio.html")

