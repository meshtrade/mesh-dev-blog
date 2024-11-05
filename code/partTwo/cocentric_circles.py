import numpy as np
import plotly.graph_objects as go
from sklearn.datasets import make_circles

# Generate concentric circles
n_samples = 1000
X, y = make_circles(n_samples=n_samples, noise=0.07, factor=0.5, random_state=42)

# Create the scatter plot
fig = go.Figure()

# Add points for each circle
colors = ['blue', 'red']  # Blue and Orange
names = ['Inner Circle', 'Outer Circle']

for i in range(2):
    mask = y == i
    fig.add_trace(go.Scatter(
        x=X[mask, 0],
        y=X[mask, 1],
        mode='markers',
        name=names[i],
        marker=dict(
            size=8,
            color=colors[i],
        )
    ))

# Update layout
fig.update_layout(
    title='Two Concentric Circles Dataset',
    xaxis_title='X',
    yaxis_title='Y',
    showlegend=True,
    plot_bgcolor='white',
    xaxis=dict(
        zeroline=True,
        zerolinewidth=1,
        zerolinecolor='lightgray',
        showgrid=True,
        gridwidth=1,
        gridcolor='lightgray'
    ),
    yaxis=dict(
        zeroline=True,
        zerolinewidth=1,
        zerolinecolor='lightgray',
        showgrid=True,
        gridwidth=1,
        gridcolor='lightgray'
    )
)

# Make sure the aspect ratio is equal
fig.update_yaxes(
    scaleanchor="x",
    scaleratio=1
)

# Show the plot
fig.write_html("cocentric_circles.html")
