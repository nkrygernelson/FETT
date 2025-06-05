# I just have to go through the subsample dict and note the results of the training.
#does this warrant having a new function for testing. 
#This is for a 4 fidelity model no scan
#Ngllbsc performance (diferent amounts of ngllbsc) vs npbe
#nscan perfromance
#nexp vs npbe
#nhse vs npbe
import matplotlib
matplotlib.use('tkagg')
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from multi_main import MultiTrainer
import csv

model_params = {
    "num_elements": 118,
    "num_fidelities": 5,
    "embedding_dim": 164,
    "fidelity_dim": 16,
    "num_blocks": 5,
    "num_heads": 10,
    "hidden_dim": 250,
    "dropout": 0.1,
    "pooling_type": "gated"
}
training_params = {
    "multi_train_split": 0.8,
    "epochs": 100,
    "batch_size": 32,
    "learning_rate": 0.001,
    "weight_decay": 1e-5,
}
mp = False
if mp:
    fidelity_map = {"pbe":0, "scan":1, "gllb-sc":2, "hse":3, "expt":4}
    fidelities_dir = "train"
    subsample_dict = {"pbe": 0.1, "scan": 1, "gllb-sc": 1, "hse": 1, "expt": 1}
else:
    fidelities_dir = "train_homemade"
    fidelity_map = {
        "GGA": 0,
        "SCANx": 0,
        "GLLBSC": 0,
        "HSE": 0,
        "EXPT": 0
        }
    subsample_dict = {"GGA": 0.5, "SCAN": 1, "GLLBSC": 1, "HSE": 1, "EXPT": 1}
subsampling_GGA = np.linspace(start=0.1, stop=1, num=10)

results_list= []
for sample_frac_gga in subsampling_GGA:
    subsample_dict["GGA"] = sample_frac_gga
    trainer = MultiTrainer(model_params=model_params, subsample_dict=subsample_dict, training_params=training_params, fidelities_dir=fidelities_dir, fidelity_map=fidelity_map)
    results, plot_paths = trainer.run_multifidelity_experiments()
    results_list.append(results)


fidelity_map = {
        "GGA": 0,
        "GLLBSC": 0,
        "HSE": 0,
        "EXPT": 0
        }
print(results_list)
header = ['Subsampling_GGA'] + [f"MAE_Fidelity_{name}" for name in fidelity_map.keys()]
data_rows = []

for i in range(len(subsampling_GGA)):
    row = [subsampling_GGA[i]]
    for fidelity_name in fidelity_map.keys():
        row.append(results_list[i][fidelity_name]["mae"])
    data_rows.append(row)

# --- Writing to CSV ---
csv_filename = "fidelity_data.csv"
with open(csv_filename, 'w', newline='') as csvfile:
    csvwriter = csv.writer(csvfile)
    csvwriter.writerow(header) # Write the header
    csvwriter.writerows(data_rows) # Write the data rows

print(f"Data saved as {csv_filename}")

plt.figure() # Create a new figure

for fidelity_name in fidelity_map.keys():
    mae_values = [results_list[i][fidelity_name]["mae"] for i in range(len(results_list))]
    plt.plot(subsampling_GGA, mae_values, marker = "o", label=f"Fidelity {fidelity_name}")

# Add legend, title, and labels (optional but good practice)
plt.legend()
plt.title("MAE vs. Subsampling GGA for Different Fidelities")
plt.xlabel("Subsampling GGA (fraction)")
plt.ylabel("Mean Absolute Error (MAE)")
plt.grid(True) # Add a grid for better readability

# Save the plot
# You can save in various formats like .png, .pdf, .svg, .jpg
plot_filename = "fidelity_plot.png"
plt.savefig(plot_filename)
print(f"Plot saved as {plot_filename}")

# Show the plot (optional, if you also want to see it interactively)
plt.show()