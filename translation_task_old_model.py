"""
Uses the original, pre-trained model to directly predict bandgaps on different fidelity levels,
serving as a baseline for the translation task.
"""

from multi_main import MultiTrainer
from preprocess_set_data import MultiFidelityPreprocessing
import torch
import os
import json
import pandas as pd
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, TensorDataset

# --- Configuration ---
collab = True # Set to True to save to Google Drive
load_path = os.path.join("data", "runs", "translate")

# --- Load Pre-trained Model ---
trained_model_path = os.path.join("models", "expanded_bg", "bg_expanded_best.pt")
model_params_path = os.path.join("models", "expanded_bg", "model_params.json")

# We need to instantiate the trainer to use its helper methods for loading
trainer = MultiTrainer(
    model_params_path=model_params_path,
    trained_model_path=trained_model_path,
    collab=collab
)
base_model = trainer.load_model()
base_model.eval()
device = trainer.device

# --- Output Directory ---
output_dir = os.path.join(trainer.save_prefix, "runs/translation_run_old_model")
os.makedirs(output_dir, exist_ok=True)

# --- Data Scaling (from original model's training data) ---
original_train_df = pd.read_csv(os.path.join("data", "runs", "expanded", "combined_train.csv"))
bg_mean = original_train_df['BG'].mean()
bg_std = original_train_df['BG'].std()

# --- Data Loading ---
# We only need the test set for this evaluation task.
test_df = pd.read_csv(os.path.join(load_path, "test_combined.csv"))

def create_dataset(df, preprocess):
    formulas = df['formula'].tolist()
    
    # Note: We are loading the raw, un-scaled bandgaps here.
    # The model will output normalized predictions, which we will then de-normalize.
    # The raw target (bg2) will be used for comparison.
    bg1 = torch.tensor(df['bg_1'].values, dtype=torch.float32)
    fid1 = torch.tensor(df['fidelity_id_1'].values, dtype=torch.long)
    bg2 = torch.tensor(df['bg_2'].values, dtype=torch.float32)
    fid2 = torch.tensor(df['fidelity_id_2'].values, dtype=torch.long)

    element_ids_list = []
    element_weights_list = []
    for formula in formulas:
        element_ids, element_weights = preprocess.formula_to_set_representation(formula)
        element_ids_list.append(element_ids)
        element_weights_list.append(element_weights)

    element_ids = torch.tensor(element_ids_list, dtype=torch.long)
    element_weights = torch.tensor(element_weights_list, dtype=torch.float32)

    return TensorDataset(element_ids, element_weights, fid1, fid2, bg1, bg2)

preprocess = MultiFidelityPreprocessing()
test_dataset = create_dataset(test_df, preprocess)
test_dataloader = DataLoader(test_dataset, batch_size=64, shuffle=False)

# --- Evaluation ---
print("\n" + "="*50)
print("Old Model Direct Prediction (Translation Baseline) Evaluation")
print(f"Output will be saved to {output_dir}")
print("="*50 + "\n")

all_predictions = []
all_targets = []
all_source_fidelities = []
all_target_fidelities = []

with torch.no_grad():
    for data in test_dataloader:
        element_ids, element_weights, fid1, fid2, bg1, bg2 = [d.to(device) for d in data]
        
        # Use the base_model to predict bg_2 directly using the target fidelity fid2
        predictions_normalized = base_model(element_ids, fid2, element_weights)
        
        # De-normalize the model's predictions
        predictions_orig = predictions_normalized.cpu().numpy() * bg_std + bg_mean
        
        # The target (bg2) from the dataloader is already in its original, un-scaled form
        targets_orig = bg2.cpu().numpy()
        
        all_predictions.extend(predictions_orig.tolist())
        all_targets.extend(targets_orig.tolist())
        all_source_fidelities.extend(fid1.cpu().numpy().tolist())
        all_target_fidelities.extend(fid2.cpu().numpy().tolist())

results_df = pd.DataFrame({
    'prediction': all_predictions,
    'target': all_targets,
    'source_fidelity': all_source_fidelities,
    'target_fidelity': all_target_fidelities
})
results_df.to_csv(os.path.join(output_dir, "predictions.csv"), index=False)

# --- Metrics and Plotting ---
fidelity_map = {"gga":0, "gga+u":1, "pbe_sol":2,"scan":3, "gllbsc":4, "hse":5,"expt":6}
inv_fidelity_map = {v: k for k, v in fidelity_map.items()}

target_fidelities = results_df['target_fidelity'].unique()
summary_metrics = {}

for target_fid in sorted(target_fidelities):
    target_fid_name = inv_fidelity_map.get(target_fid, f"Unknown_{target_fid}")
    print(f"--- Evaluating direct predictions on {target_fid_name} ---")
    
    fidelity_df = results_df[results_df['target_fidelity'] == target_fid]
    
    if not fidelity_df.empty:
        mae = mean_absolute_error(fidelity_df['target'], fidelity_df['prediction'])
        rmse = np.sqrt(mean_squared_error(fidelity_df['target'], fidelity_df['prediction']))
        r2 = r2_score(fidelity_df['target'], fidelity_df['prediction'])
        
        summary_metrics[target_fid_name] = {'mae': mae, 'rmse': rmse, 'r2': r2}
        
        print(f"  MAE: {mae:.4f}, RMSE: {rmse:.4f}, R2: {r2:.4f}")
        
        # Plotting
        plt.figure(figsize=(8, 8))
        plt.scatter(fidelity_df['target'], fidelity_df['prediction'], alpha=0.5)
        min_val = min(fidelity_df['target'].min(), fidelity_df['prediction'].min())
        max_val = max(fidelity_df['target'].max(), fidelity_df['prediction'].max())
        plt.plot([min_val, max_val], [min_val, max_val], 'r--', label='Ideal')
        plt.xlabel("Actual Bandgap (eV)")
        plt.ylabel("Predicted Bandgap (eV)")
        plt.title(f"Direct Prediction on {target_fid_name} Data")
        plt.legend()
        plt.grid(True)
        plt.savefig(os.path.join(output_dir, f"direct_prediction_on_{target_fid_name}.png"))
        plt.close()

# Summary plot
if summary_metrics:
    metrics_df = pd.DataFrame(summary_metrics).T
    metrics_df.to_csv(os.path.join(output_dir, "performance_metrics.csv"))
    metrics_df.plot(kind='bar', subplots=True, figsize=(12, 8), layout=(1, 3), legend=False)
    plt.suptitle("Direct Prediction Performance Summary by Target Fidelity")
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(os.path.join(output_dir, "summary_metrics.png"))
    plt.close()

# Overall plot
plt.figure(figsize=(8, 8))
for target_fid in sorted(target_fidelities):
    target_fid_name = inv_fidelity_map.get(target_fid, f"Unknown_{target_fid}")
    fidelity_df = results_df[results_df['target_fidelity'] == target_fid]
    if not fidelity_df.empty:
        plt.scatter(fidelity_df['target'], fidelity_df['prediction'], alpha=0.5, label=f"To {target_fid_name}")

min_val = min(results_df['target'].min(), results_df['prediction'].min())
max_val = max(results_df['target'].max(), results_df['prediction'].max())
plt.plot([min_val, max_val], [min_val, max_val], 'r--', label='Ideal')
plt.xlabel("Actual Bandgap (eV)")
plt.ylabel("Predicted Bandgap (eV)")
plt.title("Overall Direct Prediction Performance")
plt.legend()
plt.grid(True)
plt.savefig(os.path.join(output_dir, "overall_direct_prediction.png"))
plt.close()

print("\n" + "="*50)
print(f"Direct prediction plots and metrics saved to '{output_dir}' directory.")
print("="*50 + "\n")