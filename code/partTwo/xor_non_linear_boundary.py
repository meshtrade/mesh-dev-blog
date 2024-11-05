import numpy as np
import plotly.graph_objects as go

# XOR points and their labels (0 or 1 for XOR output)
points = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
labels = np.array([0, 1, 1, 0])  # XOR outputs

# Colors for the two classes
colors = ['blue' if label == 0 else 'red' for label in labels]

# Generate points for a standard parabola
x_vals = np.linspace(-5, 5, 100)
y_vals = 7 * (x_vals - 0.8)**2 - 1 # Standard parabolic curve y = ax^2 with a = 1.5

# Rotate the parabola by 45 degrees
theta = np.radians(45)  # Rotation angle in radians
cos_theta, sin_theta = np.cos(theta), np.sin(theta)
x_rotated = cos_theta * x_vals - sin_theta * y_vals
y_rotated = sin_theta * x_vals + cos_theta * y_vals

# Plotting with Plotly
fig = go.Figure()

# Add XOR points
fig.add_trace(go.Scatter(x=points[:, 0], y=points[:, 1],
                         mode='markers',
                         marker=dict(color=colors, size=12),
                         name="XOR Points"))

# Add the tilted parabolic separating curve
fig.add_trace(go.Scatter(x=x_rotated, y=y_rotated,
                         mode='lines',
                         line=dict(color='black', width=2),
                         name="Tilted Parabolic Curve"))

fig.update_layout(
    title="XOR Gate Input Separation with a Tilted Parabolic Boundary",
    xaxis_title="A",
    yaxis_title="B",
    xaxis=dict(range=[-0.5, 1.5]),
    yaxis=dict(range=[-0.5, 1.5]),
    showlegend=True
)

fig.write_html("xor_non_linear.html")
