import numpy as np
import plotly.graph_objects as go
from sklearn.datasets import make_circles

def tanh(x):
    return np.tanh(x)

# Generate concentric circles
n_samples = 1000
X, y = make_circles(n_samples=n_samples, noise=0.07, factor=0.5, random_state=42)

# Define the 3x2 transformation matrix (weights)
transformation_matrix = np.array([
    [ 0.3545, -1.5267],
    [ 1.2886,  1.1708],
    [ 1.5968, -0.4909]
])

# Apply the transformation from 2D to 3D
X_transformed = tanh(X @ transformation_matrix.T + np.array([-1.2953, -1.4793,  1.3756]))

# Create the 3D scatter plot
fig = go.Figure()

# Add points for each circle with transformed coordinates
colors = ['blue', 'red']
names = ['Inner Circle', 'Outer Circle']

for i in range(2):
    mask = y == i
    fig.add_trace(go.Scatter3d(
        x=X_transformed[mask, 0],
        y=X_transformed[mask, 1],
        z=X_transformed[mask, 2],
        mode='markers',
        name=names[i],
        marker=dict(
            size=4,
            color=colors[i],
            opacity=0.7
        )
    ))

# Update layout
fig.update_layout(
    title='Cocentric Circle Neural Network Hidden Layer Transformation',
    scene=dict(
        xaxis_title='X',
        yaxis_title='Y',
        zaxis_title='Z',
        xaxis=dict(backgroundcolor="white", gridcolor="lightgray", zerolinecolor="lightgray"),
        yaxis=dict(backgroundcolor="white", gridcolor="lightgray", zerolinecolor="lightgray"),
        zaxis=dict(backgroundcolor="white", gridcolor="lightgray", zerolinecolor="lightgray"),
    ),
    showlegend=True
)

# Show the plot
fig.write_html("cocentric_circles_transformation.html")
