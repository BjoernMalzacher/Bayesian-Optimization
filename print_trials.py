from ax.api.client import Client
from ax.api.configs import RangeParameterConfig, ChoiceParameterConfig
import pandas as pd
import numpy as np
import random
from ax.modelbridge.registry import Models
from botorch.acquisition.analytic import UpperConfidenceBound, ProbabilityOfImprovement

from ax.generation_strategy.transition_criterion import MinTrials
from ax.generation_strategy.generation_node import GenerationNode
from ax.generation_strategy.model_spec import GeneratorSpec
from ax.modelbridge.registry import Generators
import os
from ax.generation_strategy.transition_criterion import MinTrials
from ax.generation_strategy.generation_node import GenerationNode
from ax.generation_strategy.model_spec import GeneratorSpec
from ax.modelbridge.registry import Generators
from ax.generation_strategy.generation_strategy import GenerationStrategy
from ax.plot.trace import optimization_trace_single_method_plotly
import os
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go




directory_path = "output" 
html_pages = []
df_list = []
min_values =[]

# Read all CSV files in the directory
for filename in os.listdir(directory_path):
    if filename.endswith(".csv"):
        file_path = os.path.join(directory_path, filename)  
        print(file_path)      
        df = pd.read_csv(file_path)
        df_list.append(df)



for df in df_list:
    # Define combinations for contour plots
    combinations = [
        ("Kr", "Vf", "k_mean"),
        ("Ar", "Vf", "k_mean"),
        ("Kr", "Ar", "k_mean"),
        ("Rs", "Vf", "k_mean"),
    ]
    # Calculate minimum value for the k_mean column
    min_k_mean = df['k_mean'].min()
    min_values.append(min_k_mean)
    
    # Create a list to hold the figures
    figs = []

    # Loop through the combinations and create contour plots
    for x_param, y_param, z_param in combinations:
        fig = px.density_contour(df, x=x_param, y=y_param, z=z_param)
        fig.update_traces(contours_coloring="fill", contours_showlabels=True)
        figs.append(fig)

    # Create the main figure
    main_fig = go.Figure()

    # Add the scatter trace
    main_fig.add_trace(go.Scatter(
        x=df['trial_index'],
        y=df['k_mean'],
        mode='lines+markers',
        name='k_mean',
        line=dict(color='blue'),
        marker=dict(size=8),
    ))
    

    # Identify where generation_node changes
    change_indices = df.index[df['generation_node'] != df['generation_node'].shift()].tolist()

    # Add vertical lines at the change points
    for index in change_indices:
        main_fig.add_vline(x=df['trial_index'][index], line=dict(color='red', dash='dash'))

    # Update layout for the main figure
    main_fig.update_layout(
        title='k_mean vs Trial Index',
        xaxis_title='Trial Index',
        yaxis_title='k_mean',
    )
    # Create a parallel coordinates plot
    parallel_fig = px.parallel_coordinates(
    df,
    dimensions=['trial_index', 'Kr', 'Rs', 'Ar', 'Vf', 'k_mean'],
    color='trial_index',  # You can change this to 'arm_name' or any other categorical variable
    color_continuous_scale=px.colors.sequential.Viridis_r,  # Color scale
    labels={'k_mean': 'K Mean', 'trial_index':'index','Kr': 'K_r', 'Rs': 'R_s', 'Ar': 'A_r', 'Vf': 'V_f'},
    title='Parallel Coordinates Plot'
    )
    # Create a unique filename for each combined HTML file
    combined_html_file = f"html/combined_plots_{len(html_pages) + 1}.html"  # Unique filename based on the count

    # Create a single HTML file to hold all figures
    combined_html = main_fig.to_html(full_html=False)  # Start with the main figure's HTML
    combined_html += parallel_fig.to_html(full_html=False)  
    # Append each contour figure's HTML to the combined HTML
    for fig in figs:
        combined_html += fig.to_html(full_html=False)

    # Save the combined HTML to a single file
    with open(combined_html_file, 'w') as f:
        f.write(combined_html)
        
    html_pages.append(combined_html_file)

average_min_k_mean = np.mean(min_values)
dist_list = []
df = pd.read_csv("homogen_thermalcond_dataset.csv")
absolute_min = min(df['k_mean'].values)
print(min_values)
print("avg min:",average_min_k_mean)
print("absolute min:",absolute_min)
for minval in min_values:
    dist_list.append(abs(absolute_min-min_values ))
print('Avg distance:',np.mean(dist_list))
# Create the command to open all files in new tabs
command = f'"/mnt/c/Program Files/Mozilla Firefox/firefox.exe" -new-tab ' + ' '.join(f'"{file}"' for file in html_pages)
# Execute the command
os.system(command)

