import plotly.express as px
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import random
#Read the CSV file
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
unique_Kr_values = []
for sample in sample_dicts:
    if sample['Kr'] not in unique_Kr_values:
        unique_Kr_values.append(sample['Kr'])

unique_Rs_values = []
for sample in sample_dicts:
    if sample['Rs'] not in unique_Rs_values:
        unique_Rs_values.append(sample['Rs'])
        
unique_Ar_values = []
for sample in sample_dicts:
    if sample['Ar'] not in unique_Ar_values:
        unique_Ar_values.append(sample['Ar'])
unique_Vf_values = []   
for sample in sample_dicts:
    if sample['Vf'] not in unique_Vf_values:
        unique_Vf_values.append(sample['Vf'])
        
print("Unique Kr values: ", unique_Kr_values)
print("Unique Rs values: ", unique_Rs_values)
print("Unique Ar values: ", unique_Ar_values)   
print("Unique Vf values: ", unique_Vf_values)