import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Function to visualize two 3D vectors, their dot product, and the projection of v1 onto v2
def visualize_dot_product_with_projection(v1, v2):
    # Compute the dot product
    dot_prod = np.dot(v1, v2)
    
    # Compute the projection of v1 onto v2
    v2_magnitude_squared = np.dot(v2, v2)  # |v2|^2
    projection_of_v1_onto_v2 = (dot_prod / v2_magnitude_squared) * v2
    
    # Create a 3D plot
    fig = make_subplots(rows=1, cols=1, specs=[[{'type': 'scatter3d'}]])
    
    # Plot vector v1
    fig.add_trace(go.Scatter3d(
        x=[0, v1[0]], y=[0, v1[1]], z=[0, v1[2]],
        mode='lines+markers',
        name=f'<1,2,3>',
        marker=dict(size=5),
        line=dict(width=6, color='blue')
    ))
    
    # Plot vector v2
    fig.add_trace(go.Scatter3d(
        x=[0, v2[0]], y=[0, v2[1]], z=[0, v2[2]],
        mode='lines+markers',
        name='<4,5,6>',
        marker=dict(size=5),
        line=dict(width=6, color='red')
    ))
    
    # Plot the projection of v1 onto v2
    fig.add_trace(go.Scatter3d(
        x=[0, projection_of_v1_onto_v2[0]], y=[0, projection_of_v1_onto_v2[1]], z=[0, projection_of_v1_onto_v2[2]],
        mode='lines+markers',
        name='projection',
        marker=dict(size=5),
        line=dict(width=6, color='green')
    ))

    # Plot the origin
    fig.add_trace(go.Scatter3d(
        x=[0], y=[0], z=[0],
        mode='markers',
        name='Origin',
        marker=dict(size=5, color='black')
    ))
    
    # Set layout for the 3D plot
    fig.update_layout(
        scene=dict(
            xaxis_title='X',
            yaxis_title='Y',
            zaxis_title='Z',
            aspectmode='cube'
        ),
        title="3D Vector Visualization, Dot Product, and Projection",
    )
    
    # Show the plot
    fig.write_html("dot_product.html")

# Example usage:
vector1 = np.array([1, 2, 3])
vector2 = np.array([4, 5, 6])

visualize_dot_product_with_projection(vector1, vector2)
