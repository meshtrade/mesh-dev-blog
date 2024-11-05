import plotly.graph_objects as go

# XOR gate inputs 
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
    color = 'blue' if point == [1, 1] or point == [0,0] else 'red'  # Color for XOR output 1 (blue) and 0 (red)
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
    name='Decision Boundary 1'
))

fig.add_trace(go.Scatter(
    x=[0, 0.5],
    y=[0.5, 0],      
    mode='lines',
    line=dict(color='black', dash='dash'),
    name='Decision Boundary 2'
))

# Update layout
fig.update_layout(
    title="XOR Gate Input Space with Decision Boundary",
    xaxis_title="Input A",
    yaxis_title="Input B",
    xaxis=dict(range=[-0.1, 4]),
    yaxis=dict(range=[-0.1, 4]),
    showlegend=False
)

# Show the plot
fig.write_html("xor_boundary.html")
