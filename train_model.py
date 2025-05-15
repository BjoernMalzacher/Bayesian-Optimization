import pandas as pd
import numpy as np

import math
import os

from ax.service.ax_client import AxClient, ObjectiveProperties
from ax.modelbridge.generation_strategy import GenerationStep, GenerationStrategy
from ax.models.torch.botorch_modular.surrogate import SurrogateSpec
from ax.models.torch.botorch_modular.utils import ModelConfig
from plotly.offline import plot
from botorch.acquisition.analytic import UpperConfidenceBound, ProbabilityOfImprovement
from botorch.models import SingleTaskGP
import pandas as pd
from ax.plot.contour import interact_contour
import numpy as np
import plotly.graph_objects as go
from ax.plot.trace import optimization_trace_single_method_plotly
from ax.plot.slice import plot_slice, plot_slice_plotly
from ax.plot.scatter import interact_fitted
from ax.modelbridge.registry import Generators
from ax.plot.diagnostic import interact_cross_validation
from ax.modelbridge.cross_validation import cross_validate

#-----------------------------Initialization-----------------------------------------------------------------------------------------
# Read the CSV file
df = pd.read_csv("homogen_thermalcond_dataset.csv")
df_cleaned = df.dropna(subset=['Kr','Rs','Ar','Vf (%)','k_mean'])

zipped_data = list(zip(
    df_cleaned['Kr'],
    df_cleaned['Rs'],
    df_cleaned['Ar'],
    df_cleaned['Vf (%)'],
    df_cleaned['k_mean']
))
sample_dicts = [
    {
        "Kr": kr,
        "Rs": rs,
        "Ar": ar,
        "Vf": vf,
        "k_mean": k_mean
    }
    for kr, rs, ar, vf, k_mean in zipped_data
]


distance_list = []

trials_count = 40
random = 5
alpha = 0

#-----------------------------Functions-----------------------------------------------------------------------------------------
# Assume you precompute these from your sample_dicts

maxvalues = {
    "Kr": max([sample["Kr"] for sample in sample_dicts]),
    "Rs": max([sample["Rs"] for sample in sample_dicts]),
    "Ar": max([sample["Ar"] for sample in sample_dicts]),
    "Vf": max([sample["Vf"] for sample in sample_dicts]),
    }
minvalues = {   
    "Kr": min([sample["Kr"] for sample in sample_dicts]),
    "Rs": min([sample["Rs"] for sample in sample_dicts]),
    "Ar": min([sample["Ar"] for sample in sample_dicts]),
    "Vf": min([sample["Vf"] for sample in sample_dicts]),
}

def normalize(x, key):  
    return (x - minvalues[key]) / (maxvalues[key] - minvalues[key])
    

def get_nearest_sample(candidate):
    nearest_sample = None
    min_distance = float('inf')
    
    for sample in sample_dicts:
        # Distance calculation based on Kr, Rs, Ar, and Vf
        distance = np.sqrt(
            (normalize(candidate['Kr'], "Kr") - normalize(sample["Kr"], "Kr"))**2 +
            (normalize(candidate['Rs'], "Rs") - normalize(sample["Rs"], "Rs"))**2 +
            (normalize(candidate['Ar'], "Ar") - normalize(sample["Ar"], "Ar"))**2 +
            (normalize(candidate['Vf'], "Vf") - normalize(sample["Vf"], "Vf"))**2)
        
        if distance < min_distance:
            min_distance = distance 
            nearest_sample = sample
    sample_dicts.remove(nearest_sample)
    distance_list.append(min_distance)  # distance in the search speace not in the solution space
    print("Candidate:", candidate)
    print("Nearest Sample:", nearest_sample)
    return nearest_sample


#-----------------------------Generation Strategy------------------------------------------------------------------------------------------

gs = GenerationStrategy(
    steps=[
        GenerationStep(  # Initialization step
            model=Generators.SOBOL,
            num_trials=random,
            min_trials_observed=5,
        ),
        GenerationStep(  # BayesOpt step
            model=Generators.BOTORCH_MODULAR,
            # No limit on how many generator runs will be produced
            num_trials=math.ceil((trials_count-random)/2+alpha),
            model_kwargs={  # Kwargs to pass to `BoTorchModel.__init__`
                "surrogate_spec": SurrogateSpec(
                    model_configs=[ModelConfig(botorch_model_class=SingleTaskGP)]
                ),
                "botorch_acqf_class": UpperConfidenceBound, #adjustable for exploration via  beta
                "acquisition_options": {"beta": 1.0  }
            },

        ),
        GenerationStep(  # BayesOpt step
            model=Generators.BOTORCH_MODULAR,       
            num_trials=round((trials_count-random)/2-alpha),
            model_kwargs={ 
                "surrogate_spec": SurrogateSpec(
                    model_configs=[ModelConfig(botorch_model_class=SingleTaskGP)]
                ),
                "botorch_acqf_class": ProbabilityOfImprovement, #Maximzation of the current best solution
            },
        ),
    ]
)
# multiple acqu fuc starting with 'exploration' later with 'exploitation' 

#-----------------------------Experiment-----------------------------------------------------------------------------------------


# Create Ax experiment
ax_client = AxClient(generation_strategy=gs)

ax_client.create_experiment(
    name="k_mean_optimization",
    parameters=[{
        "name": "Kr",
        "type": "range",
        "bounds": [5, 100],

        "value_type": "int",
    }, {
        "name": "Rs",
        "type": "range",
        "bounds": [1e-06, 10000000000.0],
        "value_type": "float",
    },  {
        "name": "Ar",
        "type": "range",
        "bounds": [1, 6],

        "value_type": "int",
    }, {
        "name": "Vf",
        "type": "range",
        "bounds": [5,60],

        "value_type": "int",
    }],

objectives={"k_mean": ObjectiveProperties(minimize=True)}
)

for _ in range(trials_count):
    params, trial_index = ax_client.get_next_trial()
    para = get_nearest_sample(params)
    trial = ax_client.get_trial(trial_index)
    save = para.copy() 
    del save["k_mean"]
    trial.arm._parameters.update(save)
    ax_client.complete_trial(trial_index=trial_index, raw_data={"k_mean": ( para["k_mean"] , 0.0)})



#-----------------------------Plotting-----------------------------------------------------------------------------------------

aqc_func_name_list = ["ProbabilityOfImprovement", "UpperConfidenceBound beta = 1", "Random"]
    
df = ax_client.get_trials_data_frame()
df.to_csv("trials_data.csv", index=True)

zipped_data = list(zip(
    df['Kr'],
    df['Rs'],
    df['Ar'],
    df['Vf'],
    df['k_mean']
))



y_vals = df["k_mean"].values
y = np.array([distance_list])  

fig = optimization_trace_single_method_plotly(
    y=y,
    optimum=0,
    title="Optimization Trace with Strategy Changes",
    ylabel="Distance in Search Space",
)

generation_strategy = ax_client.generation_strategy
strategy_changes = []
trial_count = 0
for step in generation_strategy._steps:
    strategy_changes.append((trial_count,aqc_func_name_list.pop(-1)))
    
    if step.num_trials is None:
        break
    trial_count += step.num_trials

for idx, model in strategy_changes:  
    fig.add_vline(
        x=idx,
        line=dict(color="red", dash="dash"),
        annotation_text=f"Acqu Func.: {model}",
        annotation_position="top right"
    )
    fig.update_yaxes(tickformat=".1e")  # or ".2e", ".0e" depending on your precision needs










def calc_3d_plot(x_name = 'Ar', y_name = 'Vf'):
    df = pd.read_csv('trials_data.csv')

    x_coords = sorted(df[x_name].unique())
    y_coords = sorted(df[y_name].unique())


    pivot_df = df.pivot_table(index=y_name, columns=x_name, values='k_mean')
    pivot_df = pivot_df.reindex(index=y_coords, columns=x_coords)
    z_surface_data = pivot_df.values 



    new_fig = go.Figure(data=[go.Surface(
        x=x_coords,
        y=y_coords,
        z=z_surface_data,
        colorscale='Viridis', 
        colorbar=dict(title='k_mean'),
        contours={
            "z": {
                "show": True, "usecolormap": True, "highlightcolor": "limegreen", "project": {"z": True},
            },
            "x": {"show": True, "color": "rgba(100,100,100,0.5)", "project": {"x": True}}, 
            "y": {"show": True, "color": "rgba(100,100,100,0.5)", "project": {"y": True}}, 
        }
    )])

    new_fig.update_layout(
        title=dict(text='k_mean vs '+x_name+' vs '+y_name, x=0.5),
        autosize=True, 
        width=800, height=600,
        margin=dict(l=50, r=50, b=50, t=90), 
        scene=dict(
            xaxis_title=x_name, 
            yaxis_title=y_name,
            zaxis_title='k_mean (Permeability)',
            aspectratio=dict(x=1, y=1, z=0.7), 
            camera_eye=dict(x=1.2, y=1.2, z=0.6)
        )
    )
    return new_fig



#Unique Kr values:  [5.0, 10.0, 15.0, 25.0, 40.0, 55.0, 70.0, 100.0]
#Unique Rs values:  [10000.0, 0.0001, 1e-06, 0.01, 100.0, 1000000.0, 100000000.0, 10000000000.0]
#Unique Ar values:  
#Unique Vf values:  
# Get Ax figures (don't use render yet)
ax_figs = [
    interact_fitted(model=ax_client.generation_strategy.model, rel=False),
    interact_contour(model=ax_client.generation_strategy.model, metric_name="k_mean")
]
figs = [go.Figure(fig.data) for fig in ax_figs]

# Generate combined HTML using Plotly (Ax uses Plotly internally)
html_parts = [
    fig.to_html(), 
    calc_3d_plot(x_name='Ar', y_name='Vf').to_html(), 
    calc_3d_plot(x_name='Rs', y_name='Kr').to_html(),
    calc_3d_plot(x_name='Vf', y_name='Kr').to_html(),
    calc_3d_plot(x_name='Vf', y_name='Ar').to_html(),
    ] 

# Add the rest of the figures using list comprehension
html_parts += [
    plot(f, include_plotlyjs=(i == 0), output_type="div") for i, f in enumerate(figs)
]


html_content = f"""
<html>
<head><title>Ax Plots</title></head>
<body>
{''.join(html_parts)}
</body>
</html>
"""

# Save to a file
with open("ax_combined_report.html", "w") as f:
    f.write(html_content)

# Open in Windows Firefox (from WSL)
os.system(r"/mnt/c/Program\ Files/Mozilla\ Firefox/firefox.exe ax_combined_report.html")




# test diffrent visualization methods -
# How to use plotly


# task:
# adding points to the slice plot 
# seed testing beta -
#figure out the beta -
# change timing of exploration vs exploitation -
