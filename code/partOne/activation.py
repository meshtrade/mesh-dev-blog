import numpy as np
import plotly.graph_objects as go

# Define the step function
def step_function(S):
    return np.where(S < 0.0, 0, 1)

# Generate the S values (input range)
S = np.linspace(-1, 1, 500)

# Compute the corresponding f(S) values
f_S = step_function(S)

# Create the step plot using Plotly
fig = go.Figure()

fig.add_trace(go.Scatter(
    x=S, 
    y=f_S, 
    mode='lines',
    line_shape='hv',  # To get the step effect
    name='f(S)'
))

# Set plot title and labels
fig.update_layout(
    title="Step Function f(S)",
    xaxis_title="S",
    yaxis_title="f(S)",
    yaxis_range=[-0.2, 1.2],  # To keep some space above and below the plot
    showlegend=True
)

# Show the plot
fig.write_html("activation.html")
