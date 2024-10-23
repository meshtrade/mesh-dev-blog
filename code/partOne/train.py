import numpy as np
import plotly.graph_objects as go

# AND gate inputs and expected outputs
X = np.array([
    [0, 0],
    [0, 1],
    [1, 0],
    [1, 1]
])
y = np.array([0, 0, 0, 1])  # AND gate output

# Training parameters
learning_rate = 0.1
epochs = 40
errors = []
epochWeightsA = []
epochWeightsB = []

# Define the activation function
def step(x):
    return 1 if x > 0.0 else 0

def perceptron(input, w, bias):
    return step(np.dot(input,w) + bias)

def mse_loss(y_true, y_pred):
    return np.square(y_true - y_pred).mean()

def train_perceptron(X, y, lr, epochs):
    # Initialize weights
    weights = np.array([-1.35,1.1])
    bias = 0
    
    # Training loop
    for epoch in range(epochs):
        print(f"Epoch {epoch+1}/{epochs}")
        predictions = []
        for i in range(len(y)):
            prediction = perceptron(X[i], weights, bias)            
            predictions.append(prediction)
            if prediction != y[i]:
                weights += learning_rate * (y[i] - prediction) * X[i] 
                bias += learning_rate * (y[i] - prediction)

        print(predictions)
        errors.append(mse_loss(y, np.array(predictions)))
        epochWeightsA.append(weights[0])
        epochWeightsB.append(weights[1])
        print(f"Weights: {weights}")

    return weights, bias

# Train the perceptron
final_weights, final_bias = train_perceptron(X, y, learning_rate, epochs)

# Output final weights
print(f"\nFinal Error: {errors[-1]}")
print(f"\nFinal weights: {final_weights}")
print(f"\nBias: {final_bias}")

# Test the trained perceptron
def predict(x):
    weighted_sum = np.dot(x, final_weights) + final_bias
    return 1 if weighted_sum > 0.0 else 0

# Test on the AND gate inputs
for input_data in X:
    print(f"Input: {input_data}, Predicted Output: {predict(input_data)}")

# Create a figure
fig = go.Figure()

# Add a line plot for loss over epochs
fig.add_trace(go.Scatter(x=list(range(epochs*len(y))), 
                         y=errors, 
                         mode='lines', 
                         name='Loss'))

# Set plot title and labels
fig.update_layout(
    title="MSE Loss Over Epochs",
    xaxis_title="Epochs",
    yaxis_title="MSE Loss",
    legend=dict(
        x=0.9, y=1.1, traceorder="normal"
    ),
)

# Save the figure as an HTML file
fig.write_html("loss_and.html")

# Create first set of scatter points
scatter1 = go.Scatter(
    x=list(range(epochs)),
    y=epochWeightsA,
    mode='markers',
    name='Average Weights A',  # Legend label for the first set
)

# Create second set of scatter points
scatter2 = go.Scatter(
    x=list(range(epochs)),
    y=epochWeightsB,
    mode='markers',
    name='Average Weights B',  # Legend label for the second set
)

# Create the figure and add both scatter plots
fig = go.Figure(data=[scatter1, scatter2])

# Customize the layout
fig.update_layout(
    title='Perceptron Weights Over Epochs',
    xaxis_title='Epochs',
    yaxis_title='Weights',
    showlegend=True
)

fig.write_html("weights_and.html")

