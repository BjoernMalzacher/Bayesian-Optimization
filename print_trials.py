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
import matplotlib.pyplot as plt
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

import matplotlib.pyplot as plt
import numpy as np
import base64
from io import BytesIO


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
    '''for (x_param, y_param, z_param) in combinations:
        fig = px.density_contour(df, x=x_param, y=y_param, z=z_param)
        fig.update_traces(contours_coloring="fill", contours_showlabels=True)  # Disable legend for individual plots
        figs.append(fig)'''

    for (x_param, y_param, z_param) in combinations:

        # Erstelle ein go.Figure-Objekt
        fig = go.Figure(data=[go.Contour(
            x=df[x_param],
            y=df[y_param],
            z=df[z_param],
            colorscale='Viridis_r',  # Oder eine andere Farbskala deiner Wahl
            contours_coloring='fill', # Konturen füllen
            colorbar=dict(title=z_param) # Legendentitel setzen
        )])
        fig.update_layout(
            title=f'Konturplot von {z_param} über {x_param} und {y_param}',
            xaxis_title=x_param,
            yaxis_title=y_param
        )

        #figs.append(fig)


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

 



    iteration_id = df['trial_index'] # iteration number (x)
    score = df['k_mean'] # metric, K_mean for you (y)
    
    
    # Plot
    fig, ax = plt.subplots(figsize=(10, 8))

    # Compute the cumulative minimum
    cum_min = np.minimum.accumulate(score)

    # Plot the actual current score(/K_mean) -- in light grey, in the background
    ax.plot(iteration_id, score, 'o-', color='#cccccc', alpha=0.6, label='K_mean')

    # Plot the convergence curve (cumulative minimum error) -- overall min at each iteration
    ax.plot(iteration_id, cum_min, linestyle='-', label='Cumulative minimum error', color='tab:blue')

    # Plot threshold
    y_target = 0
    threshold = 0.03
    ax.fill_between(iteration_id, y_target, y_target + threshold, color='C1', alpha=0.15, label="Threshold")
    
    # Add a vertical line at a specific iteration (e.g., iteration 15)
        # Identify where generation_node changes
    change_indices = df.index[df['generation_node'] != df['generation_node'].shift()].tolist()

    # Add vertical lines at the change points
    for index in change_indices:    
        ax.axvline(x=index, color='red', linestyle='--', label=df['generation_node'][index])

    # Labels
    ax.set_ylabel('Error', fontsize=12)
    ax.set_xlabel('Iteration', fontsize=12)
    # Customize ticks
    ax.set_xticks(iteration_id)

    # To hide some of the tick labels in case it gets too messy
    for index, label in enumerate(ax.xaxis.get_ticklabels()):
        if index % 5 != 0:
            label.set_visible(False)

    # Customizable figure size
    plt.rcParams["figure.figsize"] = [7, 5]
    fig.subplots_adjust(hspace=0.3)

    # Highlight best iteration
    highlight_idx = df['k_mean'].idxmin()

    ax.plot(iteration_id[highlight_idx], cum_min[highlight_idx],
            marker='o', markersize=8, color='#39e75f', alpha=0.6, linestyle='None', zorder=5, label="Best result")

    # Optional legend (can be enabled if space allows)
    ax.legend(loc='best', frameon=False)

    #plt.title('Overview Bayesian Oracle Search') # default exploration/exploitation rate: 2.6
    #plt.savefig('/path/to/folder/method_convergence.pdf', dpi=300, bbox_inches='tight')
    # Convert the figure to a base64 string
    def figure_to_base64(fig):
        img = BytesIO()
        fig.savefig(img, format='png')
        img.seek(0)
        return base64.b64encode(img.getvalue()).decode('utf-8')

    # Get the base64 string
    base64_image = figure_to_base64(fig)

    # Create the HTML string
    combined_html += f'<img src="data:image/png;base64,{base64_image}" />'
    


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
print('Avg median:', np.median(dist_list))
# Create the command to open all files in new tabs
command = f'"/mnt/c/Program Files/Mozilla Firefox/firefox.exe" -new-tab ' + ' '.join(f'"{file}"' for file in html_pages)
# Execute the command
os.system(command)

