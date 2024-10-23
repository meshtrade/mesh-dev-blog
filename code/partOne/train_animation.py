import numpy as np
import plotly.graph_objs as go
from plotly.subplots import make_subplots

BIAS = -0.6

# Function to create y values for a given slope (m) and intercept (c)
def compute_y(w1, w2, b, x_values):
    # return m/ * x_values + c
    return ((-w1/w2) * x_values) - b/w2

# Parameters
x_range = np.linspace(-4, 4, 100)  # x values between -2 and 2
w1s = [-1.25, -0.85, -0.65, -0.35, -0.05, 0.15]  # List of different slopes (m)
w2s = [1.1, 1.1, 0.9, 0.8, 0.7, 0.5]  # List of different slopes (m)
biases = [0.0, 0.0, -0.1, -0.3, -0.4, -0.6]

# Initialize figure
fig = go.Figure()

# Create frames for each slope value
frames = []
for idx in range(len(w1s)):
    y_values = compute_y(w1s[idx], w2s[idx], BIAS, x_range)
    frames.append(go.Frame(data=[go.Scatter(x=x_range, y=y_values, mode='lines', line=dict(color='blue'))],
                           name=f"Slope: {idx}"))

# Add the first trace to initialize the plot
initial_y = compute_y(w1s[0], w2s[0], BIAS, x_range)
fig.add_trace(go.Scatter(x=x_range, y=initial_y, mode='lines', line=dict(color='blue'), name="Decision Boundary"))

fig.add_trace(go.Scatter(x=[0,0,1,1], y=[0,1,0,1], mode='markers', marker=dict(size=15, color="red"), name="AND Gate Inputs"))

# Update layout for animation
fig.update_layout(
    title="Animating decision boundary over sampled epochs",
    xaxis=dict(range=[-4, 4], title="x"),
    yaxis=dict(range=[-4, 4], title="y"),
    updatemenus=[dict(type="buttons", showactive=False,
                      buttons=[dict(label="Play",
                                    method="animate",
                                    args=[None, dict(frame=dict(duration=500, redraw=True), fromcurrent=True)])])]
)

# Add frames to the figure
fig.frames = frames

# Show the figure
fig.write_html("train_animation.html")
