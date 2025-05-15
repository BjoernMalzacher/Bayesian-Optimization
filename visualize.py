import plotly.express as px
import pandas as pd
import numpy as np
import random
 #Read the CSV file
df = pd.read_csv("trials_data.csv")
df_cleaned = df.dropna(subset=['Kr','Rs','Ar','Vf','distance_to_k_mean'])



zipped_data = list(zip(
    df_cleaned['Kr'],
    df_cleaned['Rs'],
    df_cleaned['Ar'],
    df_cleaned['Vf'],
    df_cleaned['distance_to_k_mean']
))
sample_dicts = [
    {
        "Kr": kr,
        "Rs": rs,
        "Ar": ar,
        "Vf": vf,
        "distance_to_k_mean": k_mean
    }
    for kr, rs, ar, vf, k_mean in zipped_data
]

df = pd.DataFrame(random.sample(sample_dicts, 25))
#df = pd.DataFrame(sample_dicts)
fig = px.parallel_coordinates(df, color="distance_to_k_mean",
                             color_continuous_scale=px.colors.diverging.balance,
                             color_continuous_midpoint=0.1
                             )
fig.show()

