from typing import Sequence
import pandas as pd
import numpy as np
import torch
import matplotlib.pyplot as plt
from torch import Tensor
import math
import os

from ax.service.ax_client import AxClient, ObjectiveProperties
from ax.utils.notebook.plotting import render
from ax.utils.report.render import render_report_elements
from ax.modelbridge.generation_strategy import GenerationStep, GenerationStrategy
from ax.modelbridge.registry import Models
from ax.models.torch.botorch_modular.model import BoTorchModel
from ax.models.torch.botorch_modular.surrogate import SurrogateSpec
from ax.models.torch.botorch_modular.utils import ModelConfig
from ax.plot.trace import optimization_trace_single_method
from plotly.offline import plot
from botorch.acquisition.acquisition import AcquisitionFunction
from botorch.acquisition.input_constructors import acqf_input_constructor, MaybeDict
from botorch.acquisition.logei import qLogExpectedImprovement, qLogNoisyExpectedImprovement
from botorch.acquisition.analytic import UpperConfidenceBound, ProbabilityOfImprovement
from botorch.models import SingleTaskGP
from botorch.models.model import Model
from botorch.utils.datasets import SupervisedDataset
from ax.plot.contour import plot_contour
from ax.plot.trace import optimization_trace_single_method
import plotly.express as px
import pandas as pd
from ax.plot.contour import interact_contour
import numpy as np
import plotly.graph_objects as go
from ax.plot.trace import optimization_trace_single_method_plotly
from ax.modelbridge.factory import get_and_fit_model
import functools
from ax.plot.slice import plot_slice
from ax.plot.scatter import interact_fitted

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
random = 10

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
            (normalize(candidate['Vf'], "Vf") - normalize(sample["Vf"], "Vf"))**2
        )
        
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
            model=Models.SOBOL,
            num_trials=random,
            min_trials_observed=5,
        ),
        GenerationStep(  # BayesOpt step
            model=Models.BOTORCH_MODULAR,
            # No limit on how many generator runs will be produced
            num_trials=math.ceil((trials_count-random)/2),
            model_kwargs={  # Kwargs to pass to `BoTorchModel.__init__`
                "surrogate_spec": SurrogateSpec(
                    model_configs=[ModelConfig(botorch_model_class=SingleTaskGP)]
                ),
                "botorch_acqf_class": UpperConfidenceBound, #adjustable for exploration via  beta
                
            },
            model_gen_kwargs={
        "acquisition_function_options": {
            "beta": 10.0,  # Adjust this value for exploration
                },
            }
        ),
        GenerationStep(  # BayesOpt step
            model=Models.BOTORCH_MODULAR,
            num_trials=round((trials_count-random)/2),
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
"""
    [{
        "name": "Kr",
        "type": "choice",
        "values": [5.0, 10.0, 15.0, 25.0,40.0,55.0,70.0,100.0],
        "is_ordered": True,
        "value_type": "float",
    }, {
        "name": "Rs",
        "type": "choice",
        "values": [10000.0, 0.0001, 1e-06, 0.01, 100.0, 1000000.0, 100000000.0, 10000000000.0],
        "is_ordered": True,
        "value_type": "float",
    }, {
        "name": "Ar",
        "type": "choice",
        "values": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
        "is_ordered": True,
        "value_type": "float",
    }, {
        "name": "Vf",
        "type": "choice",
        "values": [5.0, 10.0, 15.0, 20.0, 25.0, 30.0, 35.0, 40.0, 45.0, 50.0, 55.0, 60.0],
        "is_ordered": True,
        "value_type": "float",
    }],
"""

# Create Ax experiment
ax_client = AxClient(generation_strategy=gs)

ax_client.create_experiment(
    name="k_mean_optimization",
    parameters=[{
        "name": "Kr",
        "type": "range",
        "bounds": [5.0, 100.0],

        "value_type": "float",
    }, {
        "name": "Rs",
        "type": "range",
        "bounds": [1e-06, 10000000000.0],
        "value_type": "float",
    },  {
        "name": "Ar",
        "type": "range",
        "bounds": [1.0, 6.0],

        "value_type": "float",
    }, {
        "name": "Vf",
        "type": "range",
        "bounds": [5.0,60.0],

        "value_type": "float",
    }],
    # not range but list type

objectives={"k_mean": ObjectiveProperties(minimize=True)}
)

for i in range(trials_count):
    params, trial_index = ax_client.get_next_trial()
    para = get_nearest_sample(params)
    trial = ax_client.get_trial(trial_index)
    save = para.copy() 
    del save["k_mean"]
    gs = ax_client.generation_strategy
    trial.arm._parameters.update(save)
    ax_client.complete_trial(trial_index=trial_index, raw_data={"k_mean": ( para["k_mean"] , 0.0)})



#-----------------------------Plotting-----------------------------------------------------------------------------------------

aqc_func_name_list = ["ProbabilityOfImprovement", "UpperConfidenceBound beta = 1", "Random"]
    
df = ax_client.get_trials_data_frame()
df.to_csv("trials_data.csv", index=True)

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

fig.show()

# Get Ax figures (don't use render yet)
ax_figs = [
    fig,
    interact_fitted(model=ax_client.generation_strategy.model, rel=False),
    interact_contour(model=ax_client.generation_strategy.model, metric_name="k_mean"),
    plot_slice(model=ax_client.generation_strategy.model, metric_name="k_mean", param_name="Kr"),
    plot_slice(model=ax_client.generation_strategy.model, metric_name="k_mean", param_name="Rs"),
    plot_slice(model=ax_client.generation_strategy.model, metric_name="k_mean", param_name="Ar"),
    plot_slice(model=ax_client.generation_strategy.model, metric_name="k_mean", param_name="Vf"),
]
figs = [go.Figure(fig.data) for fig in ax_figs]

# Generate combined HTML using Plotly (Ax uses Plotly internally)
html_parts = [
    plot(fig, include_plotlyjs=i==0, output_type="div") for i, fig in enumerate(figs)
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
#figure out the beta
# testing with breakpoint