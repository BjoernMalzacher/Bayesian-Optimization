import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import base64
from io import BytesIO

def figure_to_base64(fig):
    """
    Converts a matplotlib figure to a base64 encoded PNG string.
    This is useful for embedding plots directly into HTML.
    """
    img = BytesIO()
    fig.savefig(img, format='png')
    img.seek(0)
    return base64.b64encode(img.getvalue()).decode('utf-8')


def printCompEq(list):
    min_k_mean_list = []
    aq_func = set()  
    fig, ax = plt.subplots(figsize=(10, 6))
    for df_list in list:
        aq_func = set()  
        max_iterations = max(df['trial_index'].max() for df in df_list) + 1
        k_means_at_iteration = np.full((len(df_list), max_iterations), np.nan)
        for i, df in enumerate(df_list):
            aq_func.update(df['generation_node'])
            trial_indices = df['trial_index'].astype(int).values
            k_means_at_iteration[i, trial_indices] = df['k_mean'].values
        
        mean_k_mean = np.nanmean(k_means_at_iteration, axis=0)
     
        iterations = np.arange(max_iterations)

        
        if len(aq_func) == 2 and 'Sobol' in aq_func:
                # Extract the 'searched value' using set difference
            searched_value = (aq_func - {'sobol'}).pop()
            
            print(searched_value)
        ax.plot(iterations, mean_k_mean, 'o-', label=searched_value)
      
    ax.set_title('Average Convergence with Confidence Interval')
    ax.set_xlabel('Iteration')
    ax.set_ylabel('k_mean')
    ax.legend()
    ax.grid(True)


    # Convert the figure to a base64 string
    plot_base64 = figure_to_base64(fig)
    plt.close(fig)  # Close the figure to free up memory

    # Append the HTML for the plot to the combined_html string
    plot_html = f"""
    <h2>Comparison of Mean Minimum k_mean</h2>
    <img src="data:image/png;base64,{plot_base64}" alt="Comparison Plot">
    """
    return plot_html

def loadlist(directory_path):
    df_list = []
    # Ensure the directory exists before trying to list files
    if not os.path.exists(directory_path):
        print(f"Directory not found: {directory_path}")
        return df_list

    for filename in os.listdir(directory_path):
        if filename.endswith(".csv"):
            file_path = os.path.join(directory_path, filename)
            print(f"Reading file: {file_path}")
            df = pd.read_csv(file_path)
            df_list.append(df)
    return df_list

directory_path_expImp = "output_expImp_0.015"  # Adjusted directory paths for clarity
directory_path_Upper = "output_UpperConfidence_0.015"
directory_path_PropOfImp = "output_PropImp_0.015"



df_list_expImp = loadlist(directory_path_expImp)
df_list_Upper = loadlist(directory_path_Upper)
df_list_PropOfImp = loadlist(directory_path_PropOfImp)

print(df for df in df_list_expImp)
comb_list = [df_list_expImp, df_list_Upper, df_list_PropOfImp]


os.makedirs("html", exist_ok=True)
combined_html_file = "html/combined_plots_results.html"
combined_html = ''





with open(combined_html_file, 'w') as f:
    f.write(printCompEq(comb_list))




firefox_path = "/mnt/c/Program Files/Mozilla Firefox/firefox.exe"
command = f'"{firefox_path}" -new-tab ' + combined_html_file 

try:
    os.system(command)
except Exception as e:
    print(f"Could not open browser. Please open '{combined_html_file}' manually.")
    print(f"Error: {e}")