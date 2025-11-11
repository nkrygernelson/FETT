"""
requires a trained model, and the last layer is modified for the translation task
"""

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from multi_main import MultiTrainer 
from preprocess_set_data import MultiFidelityPreprocessing
import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.data import DataLoader, Dataset, TensorDataset
import os
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# --- Configuration ---
collab = True # Set to True to save to Google Drive

load_path = os.path.join("data","runs","translate")

pooling_params = {} # num_motifs is empty in the table
model_params = {
    "embedding_dim": 172,
    "fidelity_dim": 20,
    "hidden_dim": 288,
    "num_blocks": 5,
    "num_fidelities":7,
    "num_elements":118,
    "num_heads": 16,
    "dropout": 0.17912079167770537,
    "pooling_type": "weighted",
    "pooling_params": pooling_params,
}


training_params = {
    "epochs": 50,
    "batch_size": 64,
    "learning_rate": 0.000351208662205836,
    "weight_decay": 0.000001093212600,
    "load_data": True,
    "load_path": load_path
}

fidelity_map = {"gga":0, "gga+u":1, "pbe_sol":2,"scan":3, "gllbsc":4, "hse":5,"expt":6}
subsample_dict = {"gga":1, "gga+u":1, "pbe_sol":1,"scan":1, "gllbsc":1, "hse":1,"expt":1}

trained_model_path = os.path.join("models","javier_finetuned","javier_finetuned.pt")
model_params_path = os.path.join("models","javier_finetuned","model_params.json")
trainer = MultiTrainer(model_params=model_params, subsample_dict=subsample_dict,
                       training_params=training_params,model_params_path=model_params_path,
                       property_name="bg",
                       trained_model_path=trained_model_path, 
                       fidelity_map=fidelity_map, optunize=False,collab=collab)
model = trainer.load_model()

# 1. Load the pre-trained base model
base_model = trainer.load_model()

# 2. Freeze the parameters of the base model
for param in base_model.parameters():
    param.requires_grad = False

class TranslationModel(nn.Module):
    def __init__(self, base_model, hidden_dim=256, dropout=0.2):
        super(TranslationModel, self).__init__()
        self.base_model = base_model
        
        # The input to the new head will be:
        # embedding_1 (size: embedding_dim + fidelity_dim)
        # embedding_2 (size: embedding_dim + fidelity_dim)
        # bandgap_1 (size: 1)
        if hasattr(base_model.deep_set.pooling, 'proj'):
            base_output_dim = base_model.deep_set.pooling.proj.out_features
        else:
            # For WeightedPooling, output dim is same as input dim to the pooling
            base_output_dim = base_model.deep_set.blocks[0].attention.embed_dim
        input_dim = base_output_dim * 2 + 1

        self.translation_head = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1)
        )

    def forward(self, element_ids, element_weights, fidelity_1, fidelity_2, bandgap_1):
        element_embeddings = self.base_model.element_embedding(element_ids)
        mask = (element_ids == 0)

        # Generate embedding for the first fidelity
        fidelity_embedding_1 = self.base_model.fidelity_embedding(fidelity_1)
        expanded_fidelity_1 = fidelity_embedding_1.unsqueeze(1).expand(-1, element_embeddings.size(1), -1)
        combined_embedding_1 = torch.cat([element_embeddings, expanded_fidelity_1], dim=2)
        embedding_1 = self.base_model.deep_set(
            combined_embedding_1, 
            element_weights, 
            mask=mask
        )
        
        # Generate embedding for the second fidelity
        fidelity_embedding_2 = self.base_model.fidelity_embedding(fidelity_2)
        expanded_fidelity_2 = fidelity_embedding_2.unsqueeze(1).expand(-1, element_embeddings.size(1), -1)
        combined_embedding_2 = torch.cat([element_embeddings, expanded_fidelity_2], dim=2)
        embedding_2 = self.base_model.deep_set(
            combined_embedding_2,
            element_weights, 
            mask=mask
        )

        # Concatenate the embeddings and the input bandgap
        # bandgap_1 needs to be reshaped to [batch_size, 1]
        combined_input = torch.cat([embedding_1, embedding_2, bandgap_1.unsqueeze(1)], dim=1)
        
        # Predict the second bandgap
        predicted_bandgap_2 = self.translation_head(combined_input).squeeze(-1)
        return predicted_bandgap_2

# --- Data Scaling ---
# Load the original training data to get the scaling parameters (mean and std)
original_train_df = pd.read_csv(os.path.join("data", "runs", "expanded", "combined_train.csv"))
bg_mean = original_train_df['bg'].mean()
bg_std = original_train_df['bg'].std()

def create_dataset(df, preprocess, mean, std):
    formulas = df['formula'].tolist()
    
    # Normalize bg_1 and bg_2
    bg1 = torch.tensor((df['bg_1'].values - mean) / std, dtype=torch.float32)
    fid1 = torch.tensor(df['fidelity_id_1'].values, dtype=torch.long)
    bg2 = torch.tensor((df['bg_2'].values - mean) / std, dtype=torch.float32)
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

# --- Data Loading ---
train_df = pd.read_csv(os.path.join(load_path,"train_combined.csv"))
val_df = pd.read_csv(os.path.join(load_path,"val_combined.csv"))
test_df = pd.read_csv(os.path.join(load_path,"test_combined.csv"))

preprocess = MultiFidelityPreprocessing()

train_dataset = create_dataset(train_df, preprocess, bg_mean, bg_std)
val_dataset = create_dataset(val_df, preprocess, bg_mean, bg_std)
test_dataset = create_dataset(test_df, preprocess, bg_mean, bg_std)

train_dataloader = DataLoader(train_dataset, batch_size=training_params['batch_size'], shuffle=True)
val_dataloader = DataLoader(val_dataset, batch_size=training_params['batch_size'], shuffle=False)
test_dataloader = DataLoader(test_dataset, batch_size=training_params['batch_size'], shuffle=False)

# --- Model, Loss, and Optimizer ---
device = trainer.device
translation_model = TranslationModel(base_model).to(device)

criterion = nn.MSELoss()
# Only optimize the parameters of the new head
optimizer = optim.Adam(translation_model.translation_head.parameters(), lr=training_params['learning_rate'])
scheduler = optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode='min', factor=0.5, patience=5,
)

# --- Training Loop ---
best_val_loss = float('inf')
epochs = training_params['epochs']
patience_val = 15
patience_counter = 0

for epoch in range(epochs):
    translation_model.translation_head.train() # Set only the head to train mode
    running_train_loss = 0.0
    for data in train_dataloader:
        element_ids, element_weights, fid1, fid2, bg1, bg2 = [d.to(device) for d in data]
        
        optimizer.zero_grad() 
        
        predictions = translation_model(element_ids, element_weights, fid1, fid2, bg1)
        loss = criterion(predictions, bg2)
        loss.backward()
        optimizer.step()
        
        running_train_loss += loss.item() * element_ids.size(0)
    
    epoch_train_loss = running_train_loss / len(train_dataset)

    # Validation
    translation_model.eval() # Set the whole model to eval mode
    running_val_loss = 0.0
    with torch.no_grad():
        for data in val_dataloader:
            element_ids, element_weights, fid1, fid2, bg1, bg2 = [d.to(device) for d in data]
            predictions = translation_model(element_ids, element_weights, fid1, fid2, bg1)
            loss = criterion(predictions, bg2)
            running_val_loss += loss.item() * element_ids.size(0)

    epoch_val_loss = running_val_loss / len(val_dataset)
    print(f"Epoch {epoch+1}/{epochs} | Train Loss: {epoch_train_loss:.4f} | Val Loss: {epoch_val_loss:.4f}")

    scheduler.step(epoch_val_loss)

    if epoch_val_loss < best_val_loss:
        best_val_loss = epoch_val_loss
        patience_counter = 0
        torch.save(translation_model.state_dict(), "translation_model_best.pt")
        print(f"New best model saved with validation loss: {best_val_loss:.4f}")
    else:
        patience_counter += 1
        if patience_counter >= patience_val:
            print(f"Early stopping after {epoch+1} epochs")
            break

# --- Evaluation ---
output_dir = os.path.join(trainer.save_prefix, "runs/translation_run")
os.makedirs(output_dir, exist_ok=True)

print("\n" + "="*50)
print("Translation Model Evaluation")
print(f"Output will be saved to {output_dir}")
print("="*50 + "\n")

translation_model.load_state_dict(torch.load("translation_model_best.pt", weights_only=True))
translation_model.eval() 

all_predictions = []
all_targets = []
all_fidelity_1 = []
all_fidelity_2 = []

with torch.no_grad():
    for data in test_dataloader:
        element_ids, element_weights, fid1, fid2, bg1, bg2 = [d.to(device) for d in data]
        predictions = translation_model(element_ids, element_weights, fid1, fid2, bg1)
        
        # De-normalize predictions and targets
        predictions_orig = predictions.cpu().numpy() * bg_std + bg_mean
        targets_orig = bg2.cpu().numpy() * bg_std + bg_mean
        
        all_predictions.extend(predictions_orig.tolist())
        all_targets.extend(targets_orig.tolist())
        all_fidelity_1.extend(fid1.cpu().numpy().tolist())
        all_fidelity_2.extend(fid2.cpu().numpy().tolist())

results_df = pd.DataFrame({
    'prediction': all_predictions,
    'target': all_targets,
    'source_fidelity': all_fidelity_1,
    'target_fidelity': all_fidelity_2
})
results_df.to_csv(os.path.join(output_dir, "predictions.csv"), index=False)


# --- Metrics and Plotting ---
inv_fidelity_map = {v: k for k, v in fidelity_map.items()}

# Create a list of all fidelity pairs
fidelity_pairs = results_df[['source_fidelity', 'target_fidelity']].drop_duplicates()

performance_data = []

for _, row in fidelity_pairs.iterrows():
    source_fid = row['source_fidelity']
    target_fid = row['target_fidelity']
    
    source_fid_name = inv_fidelity_map.get(source_fid, f"Unknown_{source_fid}")
    target_fid_name = inv_fidelity_map.get(target_fid, f"Unknown_{target_fid}")
    
    print(f"--- Evaluating translation from {source_fid_name} to {target_fid_name} ---")
    
    pair_df = results_df[(results_df['source_fidelity'] == source_fid) & (results_df['target_fidelity'] == target_fid)]
    
    num_samples = len(pair_df)
    
    if num_samples > 0:
        mae = mean_absolute_error(pair_df['target'], pair_df['prediction'])
        rmse = np.sqrt(mean_squared_error(pair_df['target'], pair_df['prediction']))
        r2 = r2_score(pair_df['target'], pair_df['prediction'])
        
        performance_data.append({
            'orig': source_fid_name,
            'target': target_fid_name,
            'mae': mae,
            'rmse': rmse,
            'r2': r2,
            'num_test_samples': num_samples
        })
        
        print(f"  MAE: {mae:.4f}, RMSE: {rmse:.4f}, R2: {r2:.4f}, Samples: {num_samples}")
        
        # Plotting for each pair
        plt.figure(figsize=(8, 8))
        plt.scatter(pair_df['target'], pair_df['prediction'], alpha=0.5)
        min_val = min(pair_df['target'].min(), pair_df['prediction'].min())
        max_val = max(pair_df['target'].max(), pair_df['prediction'].max())
        plt.plot([min_val, max_val], [min_val, max_val], 'r--', label='Ideal')
        plt.xlabel("Actual Bandgap (eV)")
        plt.ylabel("Predicted Bandgap (eV)")
        plt.title(f"Translation from {source_fid_name} to {target_fid_name}")
        plt.legend()
        plt.grid(True)
        plt.savefig(os.path.join(output_dir, f"translation_{source_fid_name}_to_{target_fid_name}.png"))
        plt.close()

# Create and save the performance table
performance_df = pd.DataFrame(performance_data)
performance_df = performance_df.sort_values(by=['orig', 'target']).reset_index(drop=True)
performance_df.to_csv(os.path.join(output_dir, "translation_performance_by_pair.csv"), index=False)

print("\n" + "="*50)
print("Translation Performance Table:")
print(performance_df)
print("="*50 + "\n")

# Overall plot
target_fidelities = results_df['target_fidelity'].unique()
plt.figure(figsize=(8, 8))
for target_fid in sorted(target_fidelities):
    target_fid_name = inv_fidelity_map.get(target_fid, f"Unknown_{target_fid}")
    fidelity_df = results_df[results_df['target_fidelity'] == target_fid]
    plt.scatter(fidelity_df['target'], fidelity_df['prediction'], alpha=0.5, label=f"To {target_fid_name}")

min_val = min(results_df['target'].min(), results_df['prediction'].min())
max_val = max(results_df['target'].max(), results_df['prediction'].max())
plt.plot([min_val, max_val], [min_val, max_val], 'r--', label='Ideal')
plt.xlabel("Actual Bandgap (eV)")
plt.ylabel("Predicted Bandgap (eV)")
plt.title("Overall Translation Performance")
plt.legend()
plt.grid(True)
plt.savefig(os.path.join(output_dir, "overall_translation.png"))
plt.close()

print("\n" + "="*50)
print(f"Translation plots and metrics saved to '{output_dir}' directory.")
print("="*50 + "\n")

# --- Prediction on home_sol data ---
print("\n" + "="*50)
print("Predicting on home_sol data")
print("="*50 + "\n")

home_sol_df = pd.read_csv("data/runs/home_sol/home_sol.csv")

# Prepare data for prediction
formulas = home_sol_df['formula'].tolist()
bg1_normalized = torch.tensor((home_sol_df['bg'].values - bg_mean) / bg_std, dtype=torch.float32)
fid1 = torch.full_like(bg1_normalized, 2, dtype=torch.long) # pbe_sol fidelity is 2

element_ids_list = []
element_weights_list = []
for formula in formulas:
    element_ids, element_weights = preprocess.formula_to_set_representation(formula)
    element_ids_list.append(element_ids)
    element_weights_list.append(element_weights)

element_ids = torch.tensor(element_ids_list, dtype=torch.long)
element_weights = torch.tensor(element_weights_list, dtype=torch.float32)

# --- Predict for HSE and EXPT ---
target_fidelities_to_predict = {'hse': 5, 'expt': 6}
predictions = {}
for target_name, target_fid in target_fidelities_to_predict.items():
    print(f"Predicting for {target_name}...")
    fid2 = torch.full_like(bg1_normalized, target_fid, dtype=torch.long)
    
    prediction_dataset = TensorDataset(element_ids, element_weights, fid1, fid2, bg1_normalized)
    prediction_loader = DataLoader(prediction_dataset, batch_size=training_params['batch_size'], shuffle=False)
    
    target_predictions_normalized = []
    with torch.no_grad():
        for data in prediction_loader:
            e_ids, e_weights, f1, f2, b1 = [d.to(device) for d in data]
            preds = translation_model(e_ids, e_weights, f1, f2, b1)
            target_predictions_normalized.extend(preds.cpu().numpy())
            
    predictions[f'predicted_{target_name}_bg'] = (np.array(target_predictions_normalized) * bg_std) + bg_mean

# --- Save results ---
results_to_save_df = home_sol_df.copy()
for pred_name, pred_values in predictions.items():
    results_to_save_df[pred_name] = pred_values

output_csv_path = os.path.join(output_dir, "home_sol_predictions.csv")
results_to_save_df.to_csv(output_csv_path, index=False)

print(f"Home-SOL predictions saved to {output_csv_path}")

# Also save one for each fidelity
for pred_name, pred_values in predictions.items():
    single_fidelity_df = home_sol_df.copy()
    single_fidelity_df[pred_name] = pred_values
    output_csv_path = os.path.join(output_dir, f"home_sol_predictions_{pred_name.split('_')[1]}.csv")
    single_fidelity_df.to_csv(output_csv_path, index=False)
    print(f"Home-SOL {pred_name.split('_')[1]} predictions saved to {output_csv_path}")

# --- Plotting the results ---
plt.figure(figsize=(10, 6))
plt.scatter(results_to_save_df['bg'], results_to_save_df['predicted_hse_bg'], label='Predicted HSE', alpha=0.5)
plt.scatter(results_to_save_df['bg'], results_to_save_df['predicted_expt_bg'], label='Predicted EXPT', alpha=0.5)
plt.xlabel("PBEsol Bandgap (eV)")
plt.ylabel("Predicted Bandgap (eV)")
plt.title("Translation from PBEsol to HSE and EXPT")
plt.legend()
plt.grid(True)
plot_path = os.path.join(output_dir, "home_sol_translation_plot.png")
plt.savefig(plot_path)
print(f"Translation plot saved to {plot_path}")
plt.close()
