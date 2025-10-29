"""
This script evaluates the pre-trained model on the dataset from Javier,
which is on the pbe_sol fidelity level.
"""

import pandas as pd
import os
import torch
from multi_main import MultiTrainer
from preprocess_set_data import MultiFidelityPreprocessing
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib.pyplot as plt

# --- Configuration ---
output_dir = "runs/javier_evaluation"
os.makedirs(output_dir, exist_ok=True)
fidelity_name = "pbe_sol"
fidelity_map = {"gga":0, "gga+u":1, "pbe_sol":2,"scan":3, "gllbsc":4, "hse":5,"expt":6}
fidelity_id = fidelity_map[fidelity_name]

# --- Load Pre-trained Model ---
trained_model_path = os.path.join("models", "expanded_bg", "bg_expanded_best.pt")
model_params_path = os.path.join("models", "expanded_bg", "model_params.json")

trainer = MultiTrainer(
    model_params_path=model_params_path,
    trained_model_path=trained_model_path
)
model = trainer.load_model()
model.eval()
device = trainer.device

# --- Data Scaling (from original model's training data) ---
original_train_df = pd.read_csv(os.path.join("data", "runs", "expanded", "combined_train.csv"))
bg_mean = original_train_df['bg'].mean()
bg_std = original_train_df['bg'].std()

# --- Data Loading ---
test_df = pd.read_csv(os.path.join("data", "runs", "home_sol", "home_sol.csv"))

def create_dataset(df, preprocess):
    formulas = df['formula'].tolist()
    targets = torch.tensor(df['bg'].values, dtype=torch.float32)

    element_ids_list = []
    element_weights_list = []
    for formula in formulas:
        element_ids, element_weights = preprocess.formula_to_set_representation(formula)
        element_ids_list.append(element_ids)
        element_weights_list.append(element_weights)

    element_ids = torch.tensor(element_ids_list, dtype=torch.long)
    element_weights = torch.tensor(element_weights_list, dtype=torch.float32)

    return TensorDataset(element_ids, element_weights, targets)

preprocess = MultiFidelityPreprocessing()
test_dataset = create_dataset(test_df, preprocess)
test_dataloader = DataLoader(test_dataset, batch_size=64, shuffle=False)

# --- Evaluation ---
print("\n" + "="*50)
print(f"Evaluating model on Javier's dataset ({fidelity_name})")
print(f"Output will be saved to {output_dir}")
print("="*50 + "\n")

all_predictions = []
all_targets = []

with torch.no_grad():
    for data in test_dataloader:
        element_ids, element_weights, targets = [d.to(device) for d in data]
        
        # Create a tensor for the fidelity_id, with the same batch size
        fid_tensor = torch.full((element_ids.size(0),), fidelity_id, dtype=torch.long, device=device)
        
        predictions_normalized = model(element_ids, fid_tensor, element_weights)
        
        predictions_orig = predictions_normalized.cpu().numpy() * bg_std + bg_mean
        targets_orig = targets.cpu().numpy()
        
        all_predictions.extend(predictions_orig.tolist())
        all_targets.extend(targets_orig.tolist())

results_df = pd.DataFrame({
    'prediction': all_predictions,
    'target': all_targets
})
results_df.to_csv(os.path.join(output_dir, "predictions_javier.csv"), index=False)

# --- Metrics and Plotting ---
mae = mean_absolute_error(results_df['target'], results_df['prediction'])
rmse = np.sqrt(mean_squared_error(results_df['target'], results_df['prediction']))
r2 = r2_score(results_df['target'], results_df['prediction'])

print(f"--- Evaluation Metrics for {fidelity_name} ---")
print(f"  MAE: {mae:.4f}, RMSE: {rmse:.4f}, R2: {r2:.4f}")

metrics_df = pd.DataFrame([{'mae': mae, 'rmse': rmse, 'r2': r2}], index=[fidelity_name])
metrics_df.to_csv(os.path.join(output_dir, "performance_metrics_javier.csv"))

# Plotting
plt.figure(figsize=(8, 8))
plt.scatter(results_df['target'], results_df['prediction'], alpha=0.5)
min_val = min(results_df['target'].min(), results_df['prediction'].min())
max_val = max(results_df['target'].max(), results_df['prediction'].max())
plt.plot([min_val, max_val], [min_val, max_val], 'r--', label='Ideal')
plt.xlabel("Actual Bandgap (eV)")
plt.ylabel("Predicted Bandgap (eV)")
plt.title(f"Evaluation on Javier's Dataset ({fidelity_name})")
plt.legend()
plt.grid(True)
plt.savefig(os.path.join(output_dir, "prediction_vs_actual_javier.png"))
plt.close()

print("\n" + "="*50)
print(f"Evaluation plots and metrics saved to '{output_dir}' directory.")
print("="*50 + "\n")