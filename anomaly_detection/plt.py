import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

# Data provided
data = {
    "Node Count": [4, 8, 12, 16],
    "Full Dataset Size (KB)": [95484.16, 190968.31, 285965.30, 381449.46],
    "Core Set Size (Samples)": [23900, 24437, 24772, 25177],
    "Core Set Size (KB)": [36596.88, 74838.31, 113602.84, 154012.43],
    "Compressed Core Set Size (Samples)": [478, 488, 495, 503],
    "Compressed Core Set Size (KB)": [731.94, 1494.50, 2270.04, 3076.95],
    "Compression Ratio (Core Set)": [2.61, 2.55, 2.52, 2.48],
    "Compression Ratio (Compressed Core Set)": [130.45, 127.78, 125.97, 123.97],
    "Accuracy (Core Set)": [0.9638, 0.9840, 0.9877, 0.9861],
    "Accuracy (Compressed Core Set)": [0.86, 0.79, 0.98, 0.97]
}

# Create a DataFrame
df = pd.DataFrame(data)

# Plotting the table
fig, ax = plt.subplots(figsize=(12, 8))
plt.axis('tight')
plt.axis('off')

# Add a title to the chart
plt.title('Node Comparison Metrics for Random Forest Model Compression', fontsize=16, fontweight='bold')

# Create a color palette for the table
cmap = sns.light_palette("blue", as_cmap=True)

# Create the table with formatting
table = plt.table(cellText=df.values, colLabels=df.columns, loc='center', colColours=["#87CEFA"]*len(df.columns), rowColours=["#87CEEB"]*len(df))


# Style adjustments for better presentation
table.auto_set_font_size(False)
table.set_fontsize(10)
table.scale(1.2, 1.2)

# Save the table as a PNG file
plt.savefig("Node_Comparison_Styled.png", dpi=300, bbox_inches='tight')

# Show the plot
plt.show()
