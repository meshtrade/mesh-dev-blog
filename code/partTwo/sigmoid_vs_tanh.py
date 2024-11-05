import numpy as np
import plotly.graph_objects as go

# Define the domain range
x_values = np.linspace(-5, 5, 500)

# Define the sigmoid and tanh functions and their derivatives
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    s = sigmoid(x)
    return s * (1 - s)

def tanh(x):
    return np.tanh(x)

def tanh_derivative(x):
    return 1 - np.tanh(x)**2

# Define the Mish activation function
def mish(x):
    return x * np.tanh(np.log1p(np.exp(x)))

# Numerical derivative of the Mish function 
def mish_derivative(x):
    omega = np.exp(3*x) + 4*np.exp(2*x) + (6+4*x)*np.exp(x) + 4*(1 + x)
    delta = 1 + pow((np.exp(x) + 1), 2)
    derivative = np.exp(x) * omega / pow(delta, 2)
    return derivative

# Compute function and derivative values
sigmoid_values = sigmoid(x_values)
sigmoid_deriv_values = sigmoid_derivative(x_values)
tanh_values = tanh(x_values)
tanh_deriv_values = tanh_derivative(x_values)
mish_values = mish(x_values)
mish_deriv_values = mish_derivative(x_values)

# Plotting with Plotly
fig = go.Figure()

# Add sigmoid and its derivative
fig.add_trace(go.Scatter(
    x=x_values, y=sigmoid_deriv_values,
    mode='lines', name='Sigmoid Derivative',
    line=dict(color='blue')
))
fig.add_trace(go.Scatter(
    x=x_values, y=sigmoid_values,
    mode='lines', name='Sigmoid',
    line=dict(color='blue', dash='dash')
))

# Add tanh and its derivative
fig.add_trace(go.Scatter(
    x=x_values, y=tanh_deriv_values,
    mode='lines', name='Tanh Derivative',
    line=dict(color='red')
))
fig.add_trace(go.Scatter(
    x=x_values, y=tanh_values,
    mode='lines', name='Tanh',
    line=dict(color='red', dash='dash')
))

# Add mish and its derivative
fig.add_trace(go.Scatter(
    x=x_values, y=mish_deriv_values,
    mode='lines', name='Mish Derivative',
    line=dict(color='orange')
))
fig.add_trace(go.Scatter(
    x=x_values, y=mish_values,
    mode='lines', name='Mish',
    line=dict(color='orange', dash='dash')
))

# Update layout
fig.update_layout(
    title="Sigmoid vs Tanh vs Mish",
    xaxis_title="x",
    yaxis_title="Value",
    legend_title="Functions",
    template="plotly_white"
)

fig.write_html("sigmoid_vs_tanh_vs_mish.html")
