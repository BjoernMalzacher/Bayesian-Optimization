from typing import Sequence
import pandas as pd
import numpy as np
import torch
import matplotlib.pyplot as plt
from torch import Tensor

from ax.service.ax_client import AxClient, ObjectiveProperties
from ax.utils.notebook.plotting import render
from ax.modelbridge.generation_strategy import GenerationStep, GenerationStrategy
from ax.modelbridge.registry import Models
from ax.models.torch.botorch_modular.model import BoTorchModel
from ax.models.torch.botorch_modular.surrogate import SurrogateSpec
from ax.models.torch.botorch_modular.utils import ModelConfig
from ax.plot.trace import optimization_trace_single_method

from botorch.acquisition.acquisition import AcquisitionFunction
from botorch.acquisition.input_constructors import acqf_input_constructor, MaybeDict
from botorch.acquisition.logei import qLogExpectedImprovement, qLogNoisyExpectedImprovement
from botorch.models import SingleTaskGP
from botorch.models.model import Model
from botorch.utils.datasets import SupervisedDataset
from ax.plot.contour import plot_contour
from ax.plot.trace import optimization_trace_single_method
import plotly.express as px
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from ax.plot.trace import optimization_trace_single_method_plotly
from ax.modelbridge.factory import get_and_fit_model
import functools


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
    
    return x





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
            # Which model to use for this step
            model=Models.SOBOL,
            # How many generator runs (each of which is then made a trial)
            # to produce with this step
            num_trials=5,
            # How many trials generated from this step must be `COMPLETED`
            # before the next one
            min_trials_observed=5,
        ),
        # figure out how the model is picked in the stand
        GenerationStep(  # BayesOpt step
            model=Models.BOTORCH_MODULAR,
            # No limit on how many generator runs will be produced
            num_trials=-1,
            model_kwargs={  # Kwargs to pass to `BoTorchModel.__init__`
                "surrogate_spec": SurrogateSpec(
                    model_configs=[ModelConfig(botorch_model_class=SingleTaskGP)]
                ),
                "botorch_acqf_class": qLogNoisyExpectedImprovement,
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
    # not range but list type

objectives={"k_mean": ObjectiveProperties(minimize=True)}
)

for i in range(50):
    params, trial_index = ax_client.get_next_trial()
    para = get_nearest_sample(params)
    trial = ax_client.get_trial(trial_index)
    save = para.copy() 
    del save["k_mean"]
    gs = ax_client.generation_strategy
    print([step.model for step in gs._steps])
    trial.arm._parameters.update(save)
    ax_client.complete_trial(trial_index=trial_index, raw_data={"k_mean": ( para["k_mean"] , 0.0)})



#-----------------------------Plotting-----------------------------------------------------------------------------------------
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
   
    strategy_changes.append((trial_count, step.model_kwargs["botorch_acqf_class"]))
    if step.num_trials is None:
        break
    trial_count += step.num_trials

for idx, model in strategy_changes:  
    fig.add_vline(
        x=idx,
        line=dict(color="red", dash="dash"),
        annotation_text=f"Strategy: {model}",
        annotation_position="top right"
    )
    fig.update_yaxes(tickformat=".1e")  # or ".2e", ".0e" depending on your precision needs

fig.show()


# Get contour data
# figure out which plot displayes the most information 
# k_mean vs iteration
render(ax_client.get_optimization_trace())
render(ax_client.get_contour_plot())
render(ax_client.get_contour_plot(param_x="Kr", param_y="Rs", metric_name="k_mean"))
render(ax_client.get_contour_plot(param_x="Ar", param_y="Vf", metric_name="k_mean"))
render(ax_client.get_contour_plot(param_x="Kr", param_y="Vf", metric_name="k_mean"))

# test diffrent visualization methods
# How to use plotly



# task:
# mult acqu fuc
# diffrent kind of ploting 
# k_mean vs iterations
# correct implementation of normalization
# mail with github and calendar
