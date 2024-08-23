import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import to_hex

# Data from the table
models = ['YOLOv8(base)', 'YOLOv10-n', 'YOLOv10-s', 'YOLOv10-m', 'YOLOv10-b', 'YOLOv10-l', 'YOLOv10-x']
map50 = [93.8, 87.9, 92.6, 93.1, 93.3, 93.4, 92.7]
map50_95 = [74.5, 69.6, 75.7, 75.4, 76.1, 76.8, 77.1]
f1_score = [88.5, 82.9, 88.1, 88.7, 89.1, 89.4, 89.3]
params = [3.0, 2.7, 8.0, 16.5, 20.4, 25.7, 31.6]
flops = [8.2, 8.4, 24.9, 65.6, 98.7, 129.8, 172.7]

# Metrics and their data
metrics = ['mAP@50 (%)', 'mAP@50-95 (%)', 'F1 (%)', 'Params (M)', 'FLOPs (G)']
data = {
    'mAP@50 (%)': map50,
    'mAP@50-95 (%)': map50_95,
    'F1 (%)': f1_score,
    'Params (M)': params,
    'FLOPs (G)': flops
}

# Normalize data to range [70, 90]
def normalize_to_range(data, min_range=70, max_range=90):
    min_data = np.min(data)
    max_data = np.max(data)
    return min_range + (data - min_data) * (max_range - min_range) / (max_data - min_data)

# Prepare normalized and original data
normalized_data = {}
for metric in metrics:
    if metric in ['Params (M)', 'FLOPs (G)']:
        normalized_data[metric] = normalize_to_range(np.array(data[metric]))
    else:
        normalized_data[metric] = np.array(data[metric])

# Function to convert RGB values from [0, 255] to [0, 1]
def rgb_to_normalized(rgb):
    return tuple(x / 255.0 for x in rgb)

# Function to plot data with custom colors
def plot_with_custom_colors(models, metrics, normalized_data, colors):
    plt.figure(figsize=(16, 4), dpi=600)  # Adjusted figure size and resolution for a flatter appearance

    for i, model in enumerate(models):
        model_data = [normalized_data[metric][i] for metric in metrics]
        plt.plot(metrics, model_data, marker='o', color=to_hex(colors[i]), label=model)

    plt.ylabel('Values (%)', fontsize=16)  # Increase font size
    plt.xticks(fontsize=17)  # Increase font size for x-ticks
    plt.yticks(fontsize=15)  # Increase font size for y-ticks
    plt.legend(fontsize=15)  # Increase font size for legend
    
    # Remove vertical grid lines, keep horizontal grid lines
    plt.grid(axis='y')
    
    plt.tight_layout()

    # Save high resolution image
    plt.savefig('high_res_plot.png', dpi=600)  # Save with high resolution
    plt.show()

# Define RGB colors for each model in [0, 255] range
colors_rgb = [
    (97, 108, 140),  # Custom RGB (97, 108, 140)
    (86, 140, 135),     # Green
    (178, 213, 155),     # Light Green
    (242, 222, 121),     # Yellow
    (217, 95, 24),   # Orange
    (59, 165, 149),   # Teal
    (232, 141, 47)    # Orange-Red
]

# Convert RGB values to [0, 1] range
colors_normalized = [rgb_to_normalized(color) for color in colors_rgb]

# Call the function with the custom colors
plot_with_custom_colors(models, metrics, normalized_data, colors_normalized)