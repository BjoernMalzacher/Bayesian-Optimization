from ax.api.client import Client
from ax.api.configs import ChoiceParameterConfig
import pandas as pd
import numpy as np
from botorch.acquisition.analytic import UpperConfidenceBound, ProbabilityOfImprovement,LogExpectedImprovement

from ax.generation_strategy.transition_criterion import MinTrials
from ax.generation_strategy.generation_node import GenerationNode
from ax.generation_strategy.model_spec import GeneratorSpec
from ax.modelbridge.registry import Generators
import os
from ax.core.experiment import Experiment
from ax.generation_strategy.transition_criterion import MinTrials ,TransitionCriterion
from ax.generation_strategy.generation_node import GenerationNode
from ax.generation_strategy.model_spec import GeneratorSpec
from ax.modelbridge.registry import Generators
from ax.generation_strategy.generation_strategy import GenerationStrategy
import os
from ax.core.trial import Trial
from botorch.acquisition import PosteriorStandardDeviation
from botorch.acquisition.input_constructors import acqf_input_constructor
from pyre_extensions import assert_is_instance
import datetime
import random

df_source = pd.read_csv("homogen_thermalcond_dataset.csv")
df_cleaned = df_source.dropna(subset=['Kr','Rs','Ar','Vf (%)','k_mean'])

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



class PostTransitionTrialsCriterion(TransitionCriterion):
    """
    Waits until a specified number of trials are completed from the current node.
    """

    def __init__(
        self,
        num_trials: int,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.num_trials = num_trials

    def is_met(
        self,
        experiment: Experiment,
        curr_node: GenerationNode,
    ):
        df = experiment.fetch_data().df

        if df.empty or "trial_index" not in df.columns:
            return False  # Not enough data yet

        
        # Count trials generated from this node
        trials_from_node = [
            t for t in experiment.trials.values()
            if t.deployed_name == curr_node.node_name and t.status.is_completed
        ]
        return len(trials_from_node) >= self.num_trials

    def block_continued_generation_error(
        self,
        node_name: str,
        experiment: Experiment,
        trials_from_node: set[int],
    ):
        raise RuntimeError(
            f"Generation blocked for node '{node_name}' because "
            f"{self.num_trials} trials from it have completed, but other criteria remain unmet."
        )



class KMeanThresholdCriterion(TransitionCriterion):
    """
    TransitionCriterion that triggers when the best observed `k_mean` exceeds a threshold.
    
    Args:
        k_mean_threshold: The value that `k_mean` must meet or exceed for the criterion
            to be considered met.
    """

    def __init__(
        self,
        k_mean_threshold: float,
        threshhold_count: int,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.k_mean_threshold = k_mean_threshold
        self.count = []
        self.limit = threshhold_count
    def is_met(
        self,
        experiment: Experiment,
        curr_node: GenerationNode,
    ):
        """
        Check if the best observed k_mean value meets or exceeds the threshold.
        """
        df = experiment.fetch_data().df
        
        if df.empty or "trial_index" not in df.columns or df["trial_index"].isna().all():
            return False
        
        latest_row = df.loc[df["trial_index"].idxmax()]
        latest_k_mean = latest_row["mean"]
        if latest_k_mean <= self.k_mean_threshold and  (len(self.count) == 0 or self.count.count(df["trial_index"].idxmax()) == 0):
            print(len(self.count) >= self.limit ,":",df["trial_index"].idxmax() )
            self.count.append((df["trial_index"].idxmax()))
            
        return len(self.count) >= self.limit
        
    def block_continued_generation_error(
        self,
        node_name: str,
        experiment: Experiment,
        trials_from_node: set[int],
    ):
        """
        Error raised when generation is blocked due to threshold being met but
        other criteria are not.
        """
        raise RuntimeError(
            f"Generation blocked for node '{node_name}' because k_mean >= "
            f"{self.k_mean_threshold}, but other criteria remain unmet."
        )
 
        

trial_count =120 
random_count = 12
alpha = 40
UpperConfidence_count = trial_count/2 + alpha
thrashhold = 0.03
trials = 50

for n in range(trials):
    df_source = pd.read_csv("homogen_thermalcond_dataset.csv")
    df_cleaned = df_source.dropna(subset=['Kr','Rs','Ar','Vf (%)','k_mean'])

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
    ProbabilityOfImp = GenerationNode(
        
                node_name="ProbabilityOfImp",
                model_specs=[
                    GeneratorSpec(
                        model_enum=Generators.BOTORCH_MODULAR,
                        model_kwargs={
                            "botorch_acqf_class": ProbabilityOfImprovement,
                        },  
                    ),
                ],
                transition_criteria=[
        PostTransitionTrialsCriterion(
            num_trials=120,
            transition_to="ProbabilityOfImp"
        )
    ],
            )
    
    ExpectedImp = GenerationNode(
        
                node_name="LogExpectedImprovement",
                model_specs=[
                    GeneratorSpec(
                        model_enum=Generators.BOTORCH_MODULAR,
                        model_kwargs={
                            "botorch_acqf_class": LogExpectedImprovement,
                        },  
                    ),
                ],
                transition_criteria=[
        PostTransitionTrialsCriterion(
            num_trials=120,
            transition_to="LogExpectedImprovement"
        )
    ],
            )
    UpperConfidence = GenerationNode(
                node_name="UpperConfidence",
                model_specs=[
                    GeneratorSpec(
                        model_enum=Generators.BOTORCH_MODULAR,
                        model_kwargs={
                            "botorch_acqf_class": UpperConfidenceBound,
                            "acquisition_options": {   "beta":0.5},
                        },
                    ),
                ],
                transition_criteria=[
                   PostTransitionTrialsCriterion(
            num_trials=120,
            transition_to="UpperConfidence"
        )
                ],
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
        nodes=[sobol, UpperConfidence])

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
    threshold = 0.015
    client.configure_optimization(objective=objective )
    client.set_generation_strategy(
        generation_strategy=gs,
    )
    trial_count = 120
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
        if(near_sample["k_mean"] <= threshold and client._generation_strategy._curr.node_name !=sobol.node_name):
            print("threshold reached:", near_sample["k_mean"])
            trial_count = i
            break
            
        
        
    df =  client.summarize()
    path = "./output/trial_data_"+ datetime.datetime.now().ctime() +"__"+  str(trial_count) +".csv"
    df.to_csv(path, index=True)
    
    

