import numpy as np
import plotly.graph_objects as go

# Generate a grid for the loss landscape
x = np.linspace(-5, 5, 50)
y = np.linspace(-5, 5, 50)
X, Y = np.meshgrid(x, y)
Z = X**2 + Y**2  # Parabolic loss function

# Calculate the gradient at a specific point (for example, near the edge of the plot)
point_x, point_y = 1, 1
grad_x, grad_y = 2 * point_x, 2 * point_y
point_z = point_x**2 + point_y**2

# Create the surface plot for the loss landscape
surface = go.Surface(z=Z, x=X, y=Y, colorscale='Viridis', colorbar=dict(title="Loss Value"))

# Combine the surface 
fig = go.Figure(data=[surface])

# Set plot layout
fig.update_layout(
    scene=dict(
        xaxis_title="Weight1",
        yaxis_title="Weight2",
        zaxis_title="Loss",
        camera=dict(eye=dict(x=1.5, y=1.5, z=1)),
    ),
    title="3D Loss Landscape"
)

# Show the plot
fig.write_html("gradient_loss.html")
