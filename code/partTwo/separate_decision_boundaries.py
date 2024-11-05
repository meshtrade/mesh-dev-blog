import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np

# XOR data points
points = [
    [0, 0],
    [0, 1],
    [1, 0],
    [1, 1]
]

# Second line (y = -x + 1): an alternate separation
x_line2 = np.linspace(-0.5, 1.5, 100)
y_line2 = -x_line2 + 1  # y = -x + 1

# Create a subplot with 1 row and 2 columns
fig = make_subplots(rows=1, cols=2, subplot_titles=("Decision Boundary 1", "Decision Boundary 2"))

for point in points:
    color = 'blue' if point == [1, 1] or point == [0,0] else 'red'  # Color for XOR output 1 (blue) and 0 (red)
    fig.add_trace(go.Scatter(
        x=[point[0]],
        y=[point[1]],
        mode='markers',
        marker=dict(size=15, color=color),
        showlegend=False,
        name=f"Input {point}"
    ), row=1, col=1)

# Plot the XOR points and first decision boundary in the first subplot
fig.add_trace(go.Scatter(
    x=[-0.5, 1],
    y=[1, -0.5],      
    mode='lines',
    line=dict(color='black', dash='dash'),
    name='Decision Boundary 1'
), row=1, col=1)

for point in points:
    color = 'blue' if point == [1, 1] or point == [0,0] else 'red'  # Color for XOR output 1 (blue) and 0 (red)
    fig.add_trace(go.Scatter(
        x=[point[0]],
        y=[point[1]],
        mode='markers',
        marker=dict(size=15, color=color),
        showlegend=False,
        name=f"Input {point}"
    ), row=1, col=2)

# Plot the XOR points and first decision boundary in the first subplot
fig.add_trace(go.Scatter(
    x=[0, 1.5],
    y=[1.5, 0],      
    mode='lines',
    line=dict(color='black', dash='dash'),
    name='Decision Boundary 1'
), row=1, col=2)

# Layout updates for each axis
fig.update_xaxes(range=[-0.5, 1.5], title="x1", row=1, col=1)
fig.update_yaxes(range=[-0.5, 1.5], title="x2", row=1, col=1)

fig.update_xaxes(range=[-0.5, 1.5], title="x1", row=1, col=2)
fig.update_yaxes(range=[-0.5, 1.5], title="x2", row=1, col=2)

# Layout details
fig.update_layout(title="XOR Decision Boundaries", showlegend=False)

# fig.show()
fig.write_html("separate_decision_boundaries.html")
