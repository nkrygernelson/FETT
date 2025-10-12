import os
import pandas as pd
from pymatgen.core.composition import Composition
from smact.screening import smact_validity

# Assuming these custom modules are in the same directory or Python path
from multi_main import MultiTrainer
from preprocess_set_data import MultiFidelityPreprocessing

# --- Helper Functions ---

def filter_valid_compounds(compounds):
    """Filters a list of formulas using SMACT charge-neutrality and electronegativity checks."""
    valid_compounds = []
    for comp in compounds:
        try:
            # Check if the composition is valid and passes smact_validity
            if smact_validity(Composition(comp)):
                valid_compounds.append(comp)
        except Exception:
            # Ignore formulas that cause errors in Composition or smact_validity
            continue
    return valid_compounds

def check_for_simplifcation(previous_data, ijk):
    """Checks if a new composition [i, j, k] is a simple multiple of a previous one."""
    for data in previous_data:
        k_factor = 0
        if data[0] != 0 and ijk[0] != 0:
            k_factor = ijk[0] / data[0]
        elif data[1] != 0 and ijk[1] != 0:
            k_factor = ijk[1] / data[1]
        elif data[2] != 0 and ijk[2] != 0:
            k_factor = ijk[2] / data[2]
        
        # Check if it's a consistent integer multiple greater than 1
        if k_factor > 1 and k_factor == int(k_factor):
            if all(data[idx] * k_factor == ijk[idx] for idx in range(3)):
                return True
    return False

# --- Configuration & Model Loading ---

fidelity_map = None
subsample_dict = None

## 1. Configure and load the Formation Energy (FE) model
print("Loading Formation Energy (FE) model...")
trained_model_path_fe = os.path.join("models", "FE_best", "FE_best.pt")
model_params_path_fe = os.path.join("models", "FE_best", "model_params.json")
trainer_fe = MultiTrainer(model_params=None, subsample_dict=subsample_dict,
                          training_params=None,
                          property_name="FE", # Important: Matches the dataframe column
                          model_params_path=model_params_path_fe,
                          trained_model_path=trained_model_path_fe,
                          fidelity_map=fidelity_map, optunize=False, collab=False)
model_fe = trainer_fe.load_model()
# Get mean and std from the FE training data
train_df_fe = pd.read_csv(os.path.join("data", "runs", "FE_standard", "combined_train.csv"))
mean_fe = train_df_fe["FE"].mean()
std_fe = train_df_fe["FE"].std()

## 2. Configure and load the Band Gap (BG) model
print("Loading Band Gap (BG) model...")
trained_model_path_bg = os.path.join("models", "BG_best", "BG_best.pt") # Assumed path
model_params_path_bg = os.path.join("models", "BG_best", "model_params.json") # Assumed path
trainer_bg = MultiTrainer(model_params=None, subsample_dict=subsample_dict,
                          training_params=None,
                          property_name="BG", # Important: Matches the dataframe column
                          model_params_path=model_params_path_bg,
                          trained_model_path=trained_model_path_bg,
                          fidelity_map=fidelity_map, optunize=False, collab=False)
model_bg = trainer_bg.load_model()
# Get mean and std from the BG training data
train_df_bg = pd.read_csv(os.path.join("data", "runs", "standard", "combined_train.csv")) # Assumed path
mean_bg = train_df_bg["BG"].mean()
std_bg = train_df_bg["BG"].std()

# --- Main Data Generation Loop ---

n_max = 28
elements = ["Ag","Al","Au","B","Ba","Bi","Be","Ca","Cd","Co","Cr","Cs","Cu","Fe","Ga","Ge","K","Hf","Hg","In","Ir","La","Li","Mg","Mn","Mo","Na","Nb","Ni","Os","Pb","Pd","Pt","Rb","Re","Rh","Ru","Sb","Sc","Sr","Sn","Ta","Tc","Ti","Tl","W","Y","Zr","Zn"]

# Define and create output directories
ternary_path_fe = os.path.join("/Users", "nicholaskryger-nelson", "Ternary", "FE")
ternary_path_bg = os.path.join("/Users", "nicholaskryger-nelson", "Ternary", "BG")
os.makedirs(ternary_path_fe, exist_ok=True)
os.makedirs(ternary_path_bg, exist_ok=True)

for element in elements:
    print(f"--- Processing element: {element} ---")
    
    # 1. Generate compositions
    coeff_list = []
    formulas = []
    for i in range(n_max + 1):
        for j in range(n_max + 1):
            for k in range(n_max + 1):
                if i == 0 and j == 0 and k == 0:
                    continue
                if i + j + k <= n_max:
                    if not check_for_simplifcation(coeff_list, [i, j, k]):
                        formula = f"{element}{i}P{j}S{k}"
                        formulas.append(formula)
                        coeff_list.append([i, j, k])

    print(f"Generated {len(formulas)} initial formulas.")
    formulas = filter_valid_compounds(formulas)
    print(f"Found {len(formulas)} SMACT-valid formulas.")

    if not formulas:
        print(f"No valid formulas for {element}, skipping.")
        continue
    
    target_zeros = [0] * len(formulas)
    target_high_fi = [4] * len(formulas)
    preprocess = MultiFidelityPreprocessing()
    preprocess.batch_size = len(formulas) - 1 if len(formulas) > 1 else 1

    # 2. Predict Formation Energy (FE)
    df_fe = pd.DataFrame({"formula": formulas, "fidelity": target_zeros, "FE": target_zeros})
    test_loader_fe = trainer_fe.create_test_dataloader(test_df=df_fe, preprocess=preprocess, mean=mean_fe, std=std_fe)
    _, predictions_fe, _ = trainer_fe.evaluate_on_fidelity(model_fe, test_loader=test_loader_fe, mean=mean_fe, std=std_fe)
    df_fe["FE"] = predictions_fe
    df_fe.to_csv(os.path.join(ternary_path_fe, f"{element}_FE.csv"), index=False)
    print(f"Saved FE predictions for {element}.")
    
    # 3. Predict Band Gap (BG)
    df_bg = pd.DataFrame({"formula": formulas, "fidelity": target_high_fi, "BG": target_zeros})
    test_loader_bg = trainer_bg.create_test_dataloader(test_df=df_bg, preprocess=preprocess, mean=mean_bg, std=std_bg)
    _, predictions_bg, _ = trainer_bg.evaluate_on_fidelity(model_bg, test_loader=test_loader_bg, mean=mean_bg, std=std_bg)
    df_bg["BG"] = predictions_bg
    df_bg.to_csv(os.path.join(ternary_path_bg, f"{element}_BG.csv"), index=False)
    print(f"Saved BG predictions for {element}.")

print("\nAll data generation complete.")