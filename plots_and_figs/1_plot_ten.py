import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Read CSV files
dataframe1 = pd.read_csv('resultsvn.csv')
dataframe2 = pd.read_csv('resultsvs.csv')
dataframe3 = pd.read_csv('resultsvm.csv')
dataframe4 = pd.read_csv('resultsvb.csv')
dataframe5 = pd.read_csv('resultsvl.csv')
dataframe6 = pd.read_csv('resultsvx.csv')

# Remove leading and trailing spaces from column names
dataframe1.columns = dataframe1.columns.str.strip()
dataframe2.columns = dataframe2.columns.str.strip()
dataframe3.columns = dataframe3.columns.str.strip()
dataframe4.columns = dataframe4.columns.str.strip()
dataframe5.columns = dataframe5.columns.str.strip()
dataframe6.columns = dataframe6.columns.str.strip()

# Set up plotting area: two rows and five columns of subplots
fig, axs = plt.subplots(2, 5, figsize=(20, 10))

# Define column names and subplot titles
columns = [
    'train/box_loss',
    'val/box_loss',
    'train/cls_loss',
    'val/cls_loss',
    'train/dfl_loss',
    'val/dfl_loss',
    'metrics/precision(B)',
    'metrics/recall(B)',
    'metrics/mAP50(B)',
    'metrics/mAP50-95(B)'
]
titles = [
    'Train Box Loss',
    'Val Box Loss',
    'Train Class Loss',
    'Val Class Loss',
    'Train DFL Loss',
    'Val DFL Loss',
    'Precision(B)',
    'Recall(B)',
    'mAP50(B)',
    'mAP50-95(B)'
]

# Smoothing function: moving average
def smooth(y, window_len=5):
    """Smoothing function: simple moving average"""
    y = np.array(y)  # Ensure y is a NumPy array
    if len(y) < window_len:
        return y
    return np.convolve(y, np.ones(window_len) / window_len, mode='valid')

# Plot each subplot
for i, col in enumerate(columns):
    row, col_idx = divmod(i, 5)  # Determine subplot position

    # Initialize x and y data for each dataframe
    x_data = []
    y_data = []

    # Read data from each dataframe
    for df, label, color, marker in zip(
        [dataframe1, dataframe2, dataframe3, dataframe4, dataframe5, dataframe6],
        ['YOLOv10n-PD', 'YOLOv10s-PD', 'YOLOv10m-PD', 'YOLOv10n-PD-V4', 'YOLOv10n-PD-V5', 'YOLOv10n-PD-V6'],
        ['b', 'g', 'orange', 'purple', 'cyan', 'brown'],
        ['o', 'x', '^', 's', 'D', '*']
    ):
        if col in df.columns:
            x = df['epoch']
            y = pd.to_numeric(df[col], errors='coerce')  # Convert to numeric type, set to NaN if conversion fails
            x_data.append(x)
            y_data.append(y)
        else:
            # If a column is missing in a dataframe, append empty data
            x_data.append(pd.Series([]))
            y_data.append(pd.Series([]))

    # Plot line charts for each dataframe
    for x, y, color, marker, label in zip(x_data, y_data, ['b', 'g', 'orange', 'purple', 'cyan', 'brown'], 
                                           ['o', 'x', '^', 's', 'D', '*'],
                                           ['YOLOv10n-PD', 'YOLOv10s-PD', 'YOLOv10m-PD', 
                                            'YOLOv10b-PD', 'YOLOv10l-PD', 'YOLOv10x-PD']):
        if not y.isnull().all() and len(y) > 0:  # Ensure y is not all NaN and not empty
            axs[row, col_idx].plot(x, y, color=color, marker=marker, label=label)
            smoothed_y = smooth(y, window_len=5)
            axs[row, col_idx].plot(x[:len(smoothed_y)], smoothed_y, color=color, linestyle='--')

    # Set title and labels
    axs[row, col_idx].set_title(titles[i])
    axs[row, col_idx].set_xlabel('Epoch')
    axs[row, col_idx].set_ylabel(col)
    axs[row, col_idx].legend(loc='best')
    axs[row, col_idx].grid(False)  # Remove grid

# Adjust spacing between subplots
plt.tight_layout()

# Save high-resolution image
plt.savefig('high_res_plot.png', dpi=300)  # dpi=300 increases image resolution
plt.show()
