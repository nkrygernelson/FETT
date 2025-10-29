"""
This script fine-tunes a pre-trained model on Javier's dataset (pbe_sol fidelity)
and then evaluates its performance across all fidelity levels.

It includes a toggle to switch between two modes:
1. Standard fine-tuning with a train/validation split.
2. Training on the full dataset and testing on a 30% sample of it.
"""

import pandas as pd
import os
import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
import numpy as np
import json
from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error
import matplotlib.pyplot as plt

from multi_main import MultiTrainer
from preprocess_set_data import MultiFidelityPreprocessing

# --- Configuration ---
collab = False # Set to True to save to Google Drive
freeze_base_model = False # Set to True to freeze all but the prediction head
train_on_all_homesol_data = True # If True, trains on all data. If False, splits 70/30 train/test.

finetuned_model_dir = "models/javier_finetuned"
os.makedirs(finetuned_model_dir, exist_ok=True)
finetuned_model_path = os.path.join(finetuned_model_dir, "javier_finetuned.pt")

# --- Load Pre-trained Model ---
print("\n" + "="*50)
print("Loading pre-trained model")
print("="*50 + "\n")
trained_model_path = os.path.join("models", "expanded_bg", "bg_expanded_best.pt")
model_params_path = os.path.join("models", "expanded_bg", "model_params.json")

trainer = MultiTrainer(
    model_params_path=model_params_path,
    trained_model_path=trained_model_path,
    collab=collab
)
model = trainer.load_model()
device = trainer.device

# --- Prepare Data for Fine-tuning ---
print("\n" + "="*50)
print("Preparing data for fine-tuning")
print("="*50 + "\n")
finetune_df = pd.read_csv(os.path.join("data", "runs", "home_sol", "home_sol.csv"))
fidelity_map = {"gga":0, "gga+u":1, "pbe_sol":2,"scan":3, "gllbsc":4, "hse":5,"expt":6}
fidelity_id = fidelity_map['pbe_sol']
finetune_df['fidelity'] = fidelity_id

# --- Normalization (using original training stats) ---
original_train_df = pd.read_csv(os.path.join("data", "runs", "expanded", "combined_train.csv"))
bg_mean = original_train_df['bg'].mean()
bg_std = original_train_df['bg'].std()

preprocess = MultiFidelityPreprocessing()

def create_finetune_dataloader(df, preprocess, mean, std, batch_size):
    data = []
    for _, row in df.iterrows():
        element_ids, element_weights = preprocess.formula_to_set_representation(row['formula'])
        normalized_bg = (row['bg'] - mean) / std
        data.append((element_ids, element_weights, row['fidelity'], normalized_bg))
    
    return DataLoader(data, batch_size=batch_size, shuffle=True, collate_fn=preprocess.collate_fn)

if train_on_all_homesol_data:
    print("Mode: Training on full homesol dataset.")
    train_df = finetune_df
    val_dataloader = None
    test_df = None
else:
    print("Mode: Splitting homesol dataset into 70% training and 30% testing sets.")
    train_df, test_df = train_test_split(finetune_df, test_size=0.3, random_state=42)
    train_df, val_df = train_test_split(train_df, test_size=0.1, random_state=42) # 10% of training for validation
    val_dataloader = create_finetune_dataloader(val_df, preprocess, bg_mean, bg_std, batch_size=32)

train_dataloader = create_finetune_dataloader(train_df, preprocess, bg_mean, bg_std, batch_size=32)

# --- Fine-tuning Loop ---
print("\n" + "="*50)
print("Starting fine-tuning")
print("="*50 + "\n")

if freeze_base_model:
    print("Freezing base model layers for fine-tuning...")
    for name, param in model.named_parameters():
        if not name.startswith('prediction_head'):
            param.requires_grad = False

original_lr = 0.000351208662205836
optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=original_lr / 100)
criterion = nn.MSELoss()
epochs = 20
best_val_loss = float('inf')

for epoch in range(epochs):
    model.train()
    running_train_loss = 0.0
    for data_batch in train_dataloader:
        element_ids, element_weights, fidelity_ids, bandgaps = [d.to(device) for d in data_batch]
        
        optimizer.zero_grad()
        predictions = model(element_ids, fidelity_ids, element_weights)
        loss = criterion(predictions, bandgaps)
        loss.backward()
        optimizer.step()
        running_train_loss += loss.item() * len(bandgaps)

    epoch_train_loss = running_train_loss / len(train_dataloader.dataset)

    # --- Validation Step ---
    if val_dataloader:
        model.eval()
        running_val_loss = 0.0
        with torch.no_grad():
            for data_batch in val_dataloader:
                element_ids, element_weights, fidelity_ids, bandgaps = [d.to(device) for d in data_batch]
                val_preds = model(element_ids, fidelity_ids, element_weights)
                val_loss = criterion(val_preds, bandgaps)
                running_val_loss += val_loss.item() * len(bandgaps)
        
        epoch_val_loss = running_val_loss / len(val_dataloader.dataset)
        print(f"Epoch [{epoch+1}/{epochs}] | Train Loss: {epoch_train_loss:.4f} | Val Loss: {epoch_val_loss:.4f}")

        if epoch_val_loss < best_val_loss:
            best_val_loss = epoch_val_loss
            torch.save(model.state_dict(), finetuned_model_path)
            print(f"New best fine-tuned model saved to {finetuned_model_path} with validation loss: {best_val_loss:.4f}")
    else:
        # If no validation, just print train loss and save model at the end
        print(f"Epoch [{epoch+1}/{epochs}] | Train Loss: {epoch_train_loss:.4f}")

# Save the final model if we didn't have a validation set
if not val_dataloader:
    torch.save(model.state_dict(), finetuned_model_path)
    print(f"Saved final fine-tuned model to {finetuned_model_path}")

# --- Evaluation on homesol Test Set (if applicable) ---
if not train_on_all_homesol_data:
    print("\n" + "="*50)
    print("Evaluating fine-tuned model on homesol test set (30% sample)")
    print("="*50 + "\n")
    
    test_output_dir = os.path.join("runs", "homesol_evaluation")
    os.makedirs(test_output_dir, exist_ok=True)

    test_dataloader = create_finetune_dataloader(test_df, preprocess, bg_mean, bg_std, batch_size=32)
    
    model.load_state_dict(torch.load(finetuned_model_path))
    model.eval()
    
    all_predictions, all_targets = [], []
    with torch.no_grad():
        for data_batch in test_dataloader:
            element_ids, element_weights, fidelity_ids, targets = [d.to(device) for d in data_batch]
            predictions_normalized = model(element_ids, fidelity_ids, element_weights)
            
            predictions_orig = predictions_normalized.cpu().numpy() * bg_std + bg_mean
            targets_orig = targets.cpu().numpy() * bg_std + bg_mean
            
            all_predictions.extend(predictions_orig.tolist())
            all_targets.extend(targets_orig.tolist())

    mae = mean_absolute_error(all_targets, all_predictions)
    rmse = np.sqrt(mean_squared_error(all_targets, all_predictions))
    r2 = r2_score(all_targets, all_predictions)
    print(f"--- Metrics on homesol Test Sample ---")
    print(f"  MAE: {mae:.4f}, RMSE: {rmse:.4f}, R2: {r2:.4f}")
    # Save metrics and plot
    pd.DataFrame([{'mae': mae, 'rmse': rmse, 'r2': r2}]).to_csv(os.path.join(test_output_dir, "metrics.csv"), index=False)
    print(f"metrics saved to {os.path.join(test_output_dir, 'metrics.csv')}")
    plt.figure(figsize=(8, 8))
    plt.scatter(all_predictions, all_targets, alpha=0.5)
    plt.plot([min(all_targets), max(all_targets)], [min(all_targets), max(all_targets)], 'r--')
    plt.xlabel("Actual Bandgap (eV)")
    plt.ylabel("Predicted Bandgap (eV)")
    plt.title("Evaluation on homesol Data (30% Sample)")
    plt.grid(True)
    plt.savefig(os.path.join(test_output_dir, "prediction_vs_actual.png"))
    plt.close()

# --- Final Full Evaluation ---
print("\n" + "="*50)
print("Running full evaluation with fine-tuned model")
print("="*50 + "\n")

with open(model_params_path, 'r') as f:
    model_params = json.load(f)

with open(os.path.join(finetuned_model_dir, "model_params.json"), "w") as f:
    json.dump(model_params, f, indent=4)

subsample_dict = {"gga":1, "gga+u":1, "pbe_sol":1,"scan":1, "gllbsc":1, "hse":1,"expt":1}
eval_trainer = MultiTrainer(
    model_params=model_params,
    fidelity_map=fidelity_map,
    subsample_dict=subsample_dict,
    property_name="bg",
    training_params={
        "load_data": True,
        "batch_size":32,
        "load_path": "data/runs/expanded"
    },
    model_params_path=os.path.join(finetuned_model_dir, "model_params.json"),
    trained_model_path=finetuned_model_path,
    collab=collab
)

eval_trainer.run_multifidelity_experiments()
