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
min_value = min([sample["k_mean"] for sample in sample_dicts])
distance_list = []

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
    print("Candidate:", candidate)
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

    print("Nearest Sample:", nearest_sample)
    return nearest_sample

trial_count =120 
random_count = 24
alpha = 40
UpperConfidence_count = trial_count/2 + alpha
from ax.core.trial import Trial
from pyre_extensions import assert_is_instance
import datetime
trials = 5

for n in range(trials):
    ProbabilityOfImp = GenerationNode(
                node_name="ProbabilityOfImp",
                model_specs=[
                    GeneratorSpec(
                        model_enum=Generators.BOTORCH_MODULAR,
                        model_kwargs={
                            "botorch_acqf_class": ProbabilityOfImprovement,
                            "acquisition_options": {},
                        },  
                    ),
                ],
            )

    UpperConfidence = GenerationNode(
                node_name="UpperConfidence",
                model_specs=[
                    GeneratorSpec(
                        model_enum=Generators.BOTORCH_MODULAR,
                        model_kwargs={
                            "botorch_acqf_class": UpperConfidenceBound,
                        "acquisition_options": {   "beta": 16.0},
                        },
                    ),
                ],
                transition_criteria=[
                MinTrials(
                    threshold=UpperConfidence_count,
                    transition_to=ProbabilityOfImp.node_name,
                    use_all_trials_in_exp=True,
                )  
                ]
            )

    sobol = GenerationNode(
            node_name="Sobol",
            model_specs=[
                GeneratorSpec(
                    model_enum=Generators.SOBOL,
                ),
            ],
            transition_criteria=[
                # Transition to BoTorch node once there are 5 trials on the experiment.
                MinTrials(
                    threshold=random_count,
                    transition_to=UpperConfidence.node_name,
                    use_all_trials_in_exp=True,
                )
            ]
        )



    gs = GenerationStrategy(
        name= "Custom Generation Strategy",
        nodes=[sobol, UpperConfidence, ProbabilityOfImp])

    Kr = ChoiceParameterConfig(
        name="Kr",
        values=[5.0, 10.0, 15.0, 25.0,40.0,55.0,70.0,100.0],
        is_ordered=True,
        parameter_type="float",
    )
    Rs = ChoiceParameterConfig(
        name="Rs",
        values= [10000.0, 0.0001, 1e-06, 0.01, 100.0, 1000000.0, 100000000.0, 10000000000.0],
        is_ordered=True,
        parameter_type="float",
    )
    Ar = ChoiceParameterConfig(
        name="Ar",
        values=[1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
        is_ordered=True,
        parameter_type="float",
    )
    Vf = ChoiceParameterConfig(
        name="Vf",
        values=[5.0, 10.0, 15.0, 20.0, 25.0, 30.0, 35.0, 40.0, 45.0, 50.0, 55.0, 60.0],
        is_ordered=True,
        parameter_type="float",
        
    )



    client = Client()
    client.configure_experiment( 
    name="k_mean_optimization",
    parameters=[Kr,Rs,Ar,Vf],

    )
    metric_name = "k_mean"  # this name is used during the optimization loop
    objective = f"-{metric_name}"  # minimization is specified by the negative sign

    client.configure_optimization(objective=objective)
    client.set_generation_strategy(
        generation_strategy=gs,
    )
    thrashhold = 0
    for i in range(trial_count):
        
        print("Iteration:", i)
        trial_index, parameters = client.get_next_trials(max_trials=1).popitem()
        near_sample = get_nearest_sample(parameters)
        save = near_sample.copy()
        del save["k_mean"]
        parameters = save.copy()    
        trial = assert_is_instance(client._experiment.trials[trial_index], Trial)
        trial.arm._parameters.update(save)
        client.complete_trial(trial_index= trial_index, raw_data={"k_mean": ( near_sample["k_mean"] , 0.0)})
        
        if (abs(near_sample['k_mean']-min_value))<= thrashhold and trial.generation_method_str != 'Sobol':
            print(abs(near_sample['k_mean']-min_value))
            trial_count = i
            break;
    
    
    
    df =  client.summarize()
    path = "./output/trial_data_"+ datetime.datetime.now().ctime() +"__"+  str(trial_count) +".csv"
    df.to_csv(path, index=True)
    
    

