# ==============================================================================
# IMPORTS
# ==============================================================================
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import base64
from io import BytesIO

# Import statements from the user's original code, which may not all be used.
# They are kept here for completeness.
from ax.api.client import Client
from ax.api.configs import RangeParameterConfig, ChoiceParameterConfig
from ax.modelbridge.registry import Models
from botorch.acquisition.analytic import UpperConfidenceBound, ProbabilityOfImprovement
from ax.generation_strategy.transition_criterion import MinTrials
from ax.generation_strategy.generation_node import GenerationNode
from ax.generation_strategy.model_spec import GeneratorSpec
from ax.modelbridge.registry import Generators
from ax.generation_strategy.generation_strategy import GenerationStrategy
from ax.plot.trace import optimization_trace_single_method_plotly
import random
import plotly.express as px
import plotly.graph_objects as go

# ==============================================================================
# HELPER FUNCTIONS
# ==============================================================================

def figure_to_base64(fig):
    """
    Converts a matplotlib figure to a base64 encoded PNG string.
    This is useful for embedding plots directly into HTML.
    """
    img = BytesIO()
    fig.savefig(img, format='png')
    img.seek(0)
    return base64.b64encode(img.getvalue()).decode('utf-8')

# ==============================================================================
# DATA LOADING AND PREPARATION
# ==============================================================================

# Directory where the output CSVs are stored.
directory_path = "output_0.014"

# List to store the optimization run dataframes.
df_list = []

# Global lists for storing plot data and results.
min_values = []
min_trial_list = []
best_value_indices = []
html_pages = []

# Create output directory for HTML if it doesn't exist.
os.makedirs("html", exist_ok=True)
combined_html_file = "html/combined_plots_results.html"
combined_html = ''

# Load and clean the main dataset.
df_source = pd.read_csv("homogen_thermalcond_dataset.csv")
df_cleaned = df_source.dropna(subset=['Kr', 'Rs', 'Ar', 'Vf (%)', 'k_mean'])

# Process the cleaned data to create a list of dictionaries.
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

# Count samples below specific k_mean thresholds.
count_12 = sum(1 for sample in sample_dicts if sample["k_mean"] < 0.012)
count_13 = sum(1 for sample in sample_dicts if sample["k_mean"] < 0.013)
count_14 = sum(1 for sample in sample_dicts if sample["k_mean"] < 0.014)
count_15 = sum(1 for sample in sample_dicts if sample["k_mean"] < 0.015)
count_16 = sum(1 for sample in sample_dicts if sample["k_mean"] < 0.016)
count_17 = sum(1 for sample in sample_dicts if sample["k_mean"] < 0.017)
count_2 = sum(1 for sample in sample_dicts if sample["k_mean"] < 0.02)

print("Number of samples with k_mean < 0.012:", count_12)
print("Number of samples with k_mean < 0.013:", count_13)
print("Number of samples with k_mean < 0.014:", count_14)
print("Number of samples with k_mean < 0.015:", count_15)
print("Number of samples with k_mean < 0.016:", count_16)
print("Number of samples with k_mean < 0.017:", count_17)
print("Number of samples with k_mean < 0.02:", count_2)


# Read all optimization run CSVs from the specified directory.
for filename in os.listdir(directory_path):
    if filename.endswith(".csv"):
        file_path = os.path.join(directory_path, filename)
        print(f"Reading file: {file_path}")
        df = pd.read_csv(file_path)
        df_list.append(df)

# ==============================================================================
# CONSOLIDATED DATA COLLECTION FOR ALL PLOTS
# ==============================================================================

# Single loop to collect data for all plots.
for df in df_list:
    min_idx = df['k_mean'].values.argmin()
    min_k_mean_value = df.loc[df.index[min_idx], 'k_mean']
    
    min_trial_list.append((min_idx, min_k_mean_value))
    best_value_indices.append(min_idx)
    min_values.append(min_k_mean_value)

# Unpack the list of tuples for the "Minimum k_mean per Trial" plot.
indices, k_means = zip(*min_trial_list)


# ==============================================================================
# GENERATE PLOT 1: BOXPLOT OF BEST VALUE INDICES
# ==============================================================================

combined_html += f'<h2>Distribution of Best Value Index per Trial</h2>'
fig, ax = plt.subplots(figsize=(10, 6))
ax.boxplot(best_value_indices, vert=False, patch_artist=True,
           boxprops=dict(facecolor='skyblue'),
           medianprops=dict(color='darkblue'),
           capprops=dict(color='darkblue'),
           whiskerprops=dict(color='darkblue'))

y_jitter = np.random.normal(0, 0.04, size=len(best_value_indices))
y_values = np.zeros(len(best_value_indices)) + y_jitter
ax.plot(best_value_indices, y_values, 'o', color='darkblue', alpha=0.8, zorder=3)

ax.set_title('Distribution of the Trial Index Where the Best Value Was Found', fontsize=16)
ax.set_xlabel('Trial Index of Best Value', fontsize=12)
ax.set_ylabel('')
mean_index = np.mean(best_value_indices)
median_index = np.median(best_value_indices)
ax.axvline(mean_index, color='red', linestyle='--', label=f'Mean: {mean_index:.2f}')
ax.axvline(median_index, color='blue', linestyle='--', label=f'Median: {median_index:.2f}')
ax.legend()
ax.set_ylim([-0.5, 0.5])
ax.set_yticks([])
plt.tight_layout()

base64_image = figure_to_base64(fig)
combined_html += f'<img src="data:image/png;base64,{base64_image}" />'
plt.close(fig)


# ==============================================================================
# GENERATE PLOT 2: MINIMUM K_MEAN PER TRIAL
# ==============================================================================

combined_html += f'<h2>Minimum k_mean per Trial</h2>'
fig, ax = plt.subplots(figsize=(8, 5))
ax.plot(range(len(k_means)), k_means, marker='o', linestyle='-')
ax.set_title('Minimum k_mean per Trial')
ax.set_xlabel('Trial Number')
ax.set_ylabel('Minimum k_mean')
ax.grid(True)
plt.tight_layout()

base64_image = figure_to_base64(fig)
combined_html += f'<img src="data:image/png;base64,{base64_image}" />'
plt.close(fig)


# ==============================================================================
# GENERATE PLOT 3: CONVERGENCE TRACE FOR EACH TRIAL
# ==============================================================================
def calc_every_experiment(df_list):
    for i, df in enumerate(df_list):
        iteration_id = df['trial_index']
        score = df['k_mean']
        
        fig, ax = plt.subplots(figsize=(10, 8))
        cum_min = np.minimum.accumulate(score)

        ax.plot(iteration_id, score, 'o-', color='#cccccc', alpha=0.6, label='K_mean')
        ax.plot(iteration_id, cum_min, linestyle='-', label='Cumulative minimum error', color='tab:blue')

        y_target = 0
        threshold = 0.03
        ax.fill_between(iteration_id, y_target, y_target + threshold, color='C1', alpha=0.15, label="Threshold")
        
        change_indices = df.index[df['generation_node'] != df['generation_node'].shift()].tolist()
        for index in change_indices:
            ax.axvline(x=index, color='red', linestyle='--', label=df['generation_node'][index])

        ax.set_ylabel('Error', fontsize=12)
        ax.set_xlabel('Iteration', fontsize=12)
        ax.set_xticks(iteration_id)
        for index, label in enumerate(ax.xaxis.get_ticklabels()):
            if index % 5 != 0:
                label.set_visible(False)

        plt.rcParams["figure.figsize"] = [7, 5]
        fig.subplots_adjust(hspace=0.3)

        highlight_idx = df['k_mean'].idxmin()
        ax.plot(iteration_id[highlight_idx], cum_min[highlight_idx],
                marker='o', markersize=8, color='#39e75f', alpha=0.6, linestyle='None', zorder=5, label="Best result")

        ax.legend(loc='best', frameon=False)
        
        combined_html += f'<h2>Convergence Trace for Trial {i+1}</h2>'
        base64_image = figure_to_base64(fig)
        combined_html += f'<img src="data:image/png;base64,{base64_image}" />'
        plt.close(fig)


# ==============================================================================
# GENERATE PLOT 4: CONVERGENCE TRACE FOR EACH TRIAL
# ==============================================================================


import matplotlib.pyplot as plt

# Assuming 'df_list' is your list of dataframes from all trials
# Concatenate all dataframes into a single dataframe for analysis
all_data = pd.concat(df_list, ignore_index=True)

fig, axes = plt.subplots(2, 2, figsize=(15, 10))
fig.suptitle('Distribution of Sampled Input Parameters', fontsize=16)

# Plotting histograms for each parameter
all_data['Kr'].hist(ax=axes[0, 0], bins=20, color='skyblue', edgecolor='black')
axes[0, 0].set_title('Kr')
axes[0, 0].set_xlabel('Value')
axes[0, 0].set_ylabel('Frequency')

all_data['Rs'].hist(ax=axes[0, 1], bins=20, color='lightgreen', edgecolor='black')
axes[0, 1].set_title('Rs')
axes[0, 1].set_xlabel('Value')

all_data['Ar'].hist(ax=axes[1, 0], bins=20, color='salmon', edgecolor='black')
axes[1, 0].set_title('Ar')
axes[1, 0].set_xlabel('Value')
axes[1, 0].set_ylabel('Frequency')

all_data['Vf'].hist(ax=axes[1, 1], bins=20, color='gold', edgecolor='black')
axes[1, 1].set_title('Vf')
axes[1, 1].set_xlabel('Value')

plt.tight_layout(rect=[0, 0, 1, 0.96])

base64_image = figure_to_base64(fig)
combined_html += f'<img src="data:image/png;base64,{base64_image}" />'
plt.close(fig)


# ==============================================================================
# GENERATE PLOT 5: CONVERGENCE TRACE FOR EACH TRIAL
# ==============================================================================

# Find the maximum number of iterations in any trial
max_iterations = max(df['trial_index'].max() for df in df_list) + 1

# Create a numpy array to store k_mean values at each iteration
# Initialize with NaNs to handle trials with different lengths
k_means_at_iteration = np.full((len(df_list), max_iterations), np.nan)

# Populate the array
for i, df in enumerate(df_list):
    # Ensure 'trial_index' is an integer type before using it as an index
    try:
        trial_indices = df['trial_index'].astype(int).values
        k_means_at_iteration[i, trial_indices] = df['k_mean'].values
    except ValueError as e:
        print(f"Error converting 'trial_index' to integer in trial {i}: {e}")
        print("Skipping this trial or handling as needed.")
        # You can add more robust error handling here if necessary.
        continue

# Calculate mean and standard deviation across all trials for each iteration
mean_k_mean = np.nanmean(k_means_at_iteration, axis=0)
std_k_mean = np.nanstd(k_means_at_iteration, axis=0)
iterations = np.arange(max_iterations)

# Plot the results
fig, ax = plt.subplots(figsize=(10, 6))
ax.plot(iterations, mean_k_mean, 'o-', label='Mean k_mean', color='tab:blue')
ax.fill_between(iterations, mean_k_mean - std_k_mean, mean_k_mean + std_k_mean, color='tab:blue', alpha=0.2, label='Standard Deviation')

ax.set_title('Average Convergence with Confidence Interval')
ax.set_xlabel('Iteration')
ax.set_ylabel('k_mean')
ax.legend()
ax.grid(True)

base64_image = figure_to_base64(fig)
combined_html += f'<img src="data:image/png;base64,{base64_image}" />'
plt.close(fig)

# ==============================================================================
# GENERATE PLOT 4: CONVERGENCE TRACE FOR EACH TRIAL
# ==============================================================================

# Concatenate all dataframes for analysis
all_data = pd.concat(df_list, ignore_index=True)

# Select the parameters you want to plot
columns_to_plot = ['Kr', 'Rs', 'Ar', 'Vf', 'k_mean']

# Correct line: Assign the returned array of axes to a single variable.
axes = pd.plotting.scatter_matrix(
    all_data[columns_to_plot], 
    figsize=(15, 15), 
    marker='o',
    hist_kwds={'bins': 20},
    diagonal='hist',
    alpha=0.8,
)
plt.suptitle('Pair Plot of Input Parameters and Output Metric', y=1.02)

# To get the Figure object, you can access it from one of the axes in the returned array.
# For example, using the first axis in the grid:
fig = axes[0, 0].get_figure()

# Now your existing code for embedding the plot will work correctly
base64_image = figure_to_base64(fig)
combined_html += f'<img src="data:image/png;base64,{base64_image}" />'

# Close the figure to free up memory
plt.close(fig)

# ==============================================================================
# GENERATE PLOT 4: CONVERGENCE TRACE FOR EACH TRIAL
# ==============================================================================

# Find the best and worst performing trials
min_k_means_by_trial = [df['k_mean'].min() for df in df_list]
best_trial_index = np.argmin(min_k_means_by_trial)
worst_trial_index = np.argmax(min_k_means_by_trial)

best_df = df_list[best_trial_index]
worst_df = df_list[worst_trial_index]

fig, ax = plt.subplots(figsize=(12, 7))

# Plot the best trial's convergence curve
best_cum_min = np.minimum.accumulate(best_df['k_mean'])
ax.plot(best_df['trial_index'], best_cum_min, label=f'Best Trial (min: {best_df["k_mean"].min():.4f})', 
        color='green', marker='o', linestyle='-')

# Plot the worst trial's convergence curve
worst_cum_min = np.minimum.accumulate(worst_df['k_mean'])
ax.plot(worst_df['trial_index'], worst_cum_min, label=f'Worst Trial (min: {worst_df["k_mean"].min():.4f})',
        color='red', marker='o', linestyle='-')

ax.set_title('Best vs. Worst Trial Convergence Comparison', fontsize=16)
ax.set_xlabel('Iteration')
ax.set_ylabel('Cumulative Minimum k_mean')
ax.legend()
ax.grid(True)


base64_image = figure_to_base64(fig)
combined_html += f'<img src="data:image/png;base64,{base64_image}" />'
plt.close(fig)

# ==============================================================================
# GENERATE PLOT 4: CONVERGENCE TRACE FOR EACH TRIAL
# ==============================================================================


# Concatenate all dataframes for analysis
all_data = pd.concat(df_list, ignore_index=True)

# Select the numerical columns to calculate correlations
columns_to_correlate = ['Kr', 'Rs', 'Ar', 'Vf', 'k_mean']
corr_matrix = all_data[columns_to_correlate].corr()

fig, ax = plt.subplots(figsize=(10, 8))
cax = ax.imshow(corr_matrix, cmap='coolwarm', vmin=-1, vmax=1)

# Add titles and labels
ax.set_title('Correlation Matrix Heatmap', fontsize=16)
ax.set_xticks(np.arange(len(columns_to_correlate)))
ax.set_yticks(np.arange(len(columns_to_correlate)))
ax.set_xticklabels(columns_to_correlate, rotation=45, ha='right')
ax.set_yticklabels(columns_to_correlate)

# Add correlation values to the heatmap
for i in range(len(columns_to_correlate)):
    for j in range(len(columns_to_correlate)):
        text = ax.text(j, i, f'{corr_matrix.iloc[i, j]:.2f}',
                       ha='center', va='center', color='black')

# Add a color bar
cbar = fig.colorbar(cax)
cbar.set_label('Correlation Coefficient')

plt.tight_layout()

base64_image = figure_to_base64(fig)
combined_html += f'<img src="data:image/png;base64,{base64_image}" />'
plt.close(fig)



# ==============================================================================
# RESULTS AND OUTPUT
# ==============================================================================

def calculate_average_tries(filename="homogen_thermalcond_dataset.csv", threshold=0.012):
    """
    Calculates the average number of random tries (Sobol) needed to get a
    value below a specified k_mean threshold.

    Args:
        filename (str): The name of the CSV file containing the data.
        threshold (float): The k_mean value to use as the success criterion.
    """
    try:
        # Load and clean the main dataset, dropping rows with any NaN values
        df_source = pd.read_csv(filename)
        df_cleaned = df_source.dropna(subset=['Kr', 'Rs', 'Ar', 'Vf (%)', 'k_mean'])

        # Get the total number of samples in the cleaned dataset
        total_samples = len(df_cleaned)

        # Count the number of samples that meet the success criterion
        successful_samples = df_cleaned[df_cleaned['k_mean'] < threshold]
        successful_count = len(successful_samples)

        # Calculate the probability of a "successful" pick in a single random try
        if total_samples > 0:
            probability_of_success = successful_count / total_samples
        else:
            print("The cleaned dataset is empty, so the calculation cannot be performed.")
            return

        # The number of trials to get the first success in a series of random
        # picks follows a geometric distribution. The average number of tries
        # is 1 divided by the probability of success.
        if probability_of_success > 0:
            average_tries = 1 / probability_of_success
            print(f"Total samples in the dataset: {total_samples}")
            print(f"Number of samples with k_mean < {threshold}: {successful_count}")
            print(f"Probability of picking a successful sample in one try: {probability_of_success:.4f}")
            print(f"\nOn average, it would take approximately {average_tries:.2f} random tries to get a value below {threshold}.")
        else:
            print(f"No samples with a k_mean value below {threshold} were found in the dataset.")
            print("The average number of tries is considered infinite.")


    except FileNotFoundError:
        print(f"Error: The file '{filename}' was not found. Please ensure the file is in the same directory as the script.")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")

# Run the function
calculate_average_tries()
calculate_average_tries(threshold=0.013)
calculate_average_tries(threshold=0.014)
calculate_average_tries(threshold=0.015)

calculate_average_tries(threshold=0.016)

calculate_average_tries(threshold=0.017)

calculate_average_tries(threshold=0.02)
# ==============================================================================
# RESULTS AND OUTPUT
# ==============================================================================

with open(combined_html_file, 'w') as f:
    f.write(combined_html)

html_pages.append(combined_html_file)

average_min_k_mean = np.mean(min_values)
df_all_data = pd.read_csv("homogen_thermalcond_dataset.csv")
absolute_min = min(df_all_data['k_mean'].values)

print("\n--- Summary Statistics ---")
print(f"Number of trials: {len(min_values)}")
#print(f"Minimum k_mean values from each trial: {min_values}")
print(f"Average minimum k_mean across all trials: {average_min_k_mean}")
print(f"Absolute minimum k_mean in the entire dataset: {absolute_min}")

dist_list = [abs(absolute_min - minval) for minval in min_values]
print(f"Average distance of trial minimums from absolute minimum: {np.mean(dist_list)}")
print(f"Median distance of trial minimums from absolute minimum: {np.median(dist_list)}")

firefox_path = "/mnt/c/Program Files/Mozilla Firefox/firefox.exe"
command = f'"{firefox_path}" -new-tab ' + ' '.join(f'"{file}"' for file in html_pages)

try:
    os.system(command)
except Exception as e:
    print(f"Could not open browser. Please open '{combined_html_file}' manually.")
    print(f"Error: {e}")