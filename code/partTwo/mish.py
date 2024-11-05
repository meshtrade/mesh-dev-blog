import numpy as np
import plotly.graph_objs as go

# Define the Mish activation function
def mish(x):
    return x * np.tanh(np.log1p(np.exp(x)))

# Numerical derivative of the Mish function using central difference
def mish_derivative(x):
    omega = np.exp(3*x) + 4*np.exp(2*x) + (6+4*x)*np.exp(x) + 4*(1 + x)
    delta = 1 + pow((np.exp(x) + 1), 2)
    derivative = np.exp(x) * omega / pow(delta, 2)
    return derivative

# Generate x values and corresponding Mish and derivative values
x_values = np.linspace(-10, 10, 500)
y_values = mish(x_values)
y_derivative_values = mish_derivative(x_values)

# Create the Plotly plot
fig = go.Figure()

# Plot Mish function
fig.add_trace(go.Scatter(
    x=x_values, 
    y=y_values, 
    mode='lines', 
    line=dict(color='red'), 
    name='Mish(x)'
))

# Plot derivative of Mish function
fig.add_trace(go.Scatter(
    x=x_values, 
    y=y_derivative_values, 
    mode='lines', 
    line=dict(color='blue', dash='dash'), 
    name="Mish'(x)"
))

# Customize the layout
fig.update_layout(
    title='Mish Activation Function and Its Derivative',
    xaxis_title='x',
    yaxis_title='y',
)

# Show the plot
fig.write_html("mish.html")
