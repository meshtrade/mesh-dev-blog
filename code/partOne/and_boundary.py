import plotly.graph_objects as go

# AND gate inputs (possible combinations of binary inputs)
points = [
    [0, 0],
    [0, 1],
    [1, 0],
    [1, 1]
]

# Create figure
fig = go.Figure()

# Plot the points
for point in points:
    color = 'blue' if point == [1, 1] else 'red'  # Color for AND output 1 (blue) and 0 (red)
    fig.add_trace(go.Scatter(
        x=[point[0]],
        y=[point[1]],
        mode='markers',
        marker=dict(size=15, color=color),
        name=f"Input {point}"
    ))

fig.add_trace(go.Scatter(
    x=[0, 2],
    y=[1.5, 0],      
    mode='lines',
    line=dict(color='black', dash='dash'),
    name='Decision Boundary'
))

# Update layout
fig.update_layout(
    title="AND Gate Input Space with Decision Boundary",
    xaxis_title="Input A",
    yaxis_title="Input B",
    xaxis=dict(range=[-0.1, 4]),
    yaxis=dict(range=[-0.1, 4]),
    showlegend=False
)

# Show the plot
fig.write_html("and_boundary_manual.html")
