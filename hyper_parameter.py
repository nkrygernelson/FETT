import optuna

from multi_main import MultiTrainer


#subsample_dict = {"pbe": 0.1, "scan": 1, "gllb-sc": 1, "hse": 1, "expt": 1}



def objective(trial):
    embedding_dim = trial.suggest_int("embedding_dim", 60, 250, step=4) 
    fidelity_dim = trial.suggest_int("fidelity_dim", 10, 20, step=2) 
    num_heads = trial.suggest_categorical("num_heads", [4, 5, 8, 10, 16]) # Common head counts
    pooling_type = trial.suggest_categorical("pooling_type", ["gated", "weighted", "hierarchical", "cross_attention", "attention"])
    pooling_params = {} # Initialize
    if pooling_type == "cross_attention":
        pooling_params["cross_attention"] = {
            "num_queries": trial.suggest_int("num_queries", 2, 10, step=1)
        }
    elif pooling_type == "hierarchical":
        pooling_params["hierarchical"] = {
            "num_motifs": trial.suggest_int("num_motifs", 2, 10, step=1)
        }
    model_params = {
        "num_elements": 118,
        "num_fidelities": 5, # Should match the number of fidelities in fidelity_map
        "embedding_dim": embedding_dim,
        "fidelity_dim": fidelity_dim,
        "num_blocks": trial.suggest_int("num_blocks", 3, 6),
        "num_heads": num_heads,
        "hidden_dim": trial.suggest_int("hidden_dim", 128, 384, step=32),
        "dropout": trial.suggest_float("dropout", 0.05, 0.3),
        "pooling_type": pooling_type,
        'pooling_params': pooling_params
    }
    if (model_params['embedding_dim'] + model_params['fidelity_dim']) % model_params['num_heads'] != 0:
        # If the condition is not met, prune this trial early.
        # Optuna will try different hyperparameter combinations.
        raise optuna.exceptions.TrialPruned(
            f"Skipping trial: (embedding_dim {model_params['embedding_dim']}+ {model_params['fidelity_dim']} fidelity_dim )"
            f"is not divisible by num_heads "
        )
    training_params = {
        "multi_train_split": 0.8,
        "epochs": trial.suggest_int('epochs',60,150, step=10),
        "batch_size": trial.suggest_categorical("batch_size", [32, 64]),
        "learning_rate": trial.suggest_float("learning_rate", 1e-4, 1e-2, log=True),
        "weight_decay": trial.suggest_float("weight_decay", 1e-6, 1e-4, log=True),
        "trial":trial,
        "load_data":True
    }

    mp = True
    if mp:
        fidelity_map = {"pbe":0, "scan":1, "gllb-sc":2, "hse":3, "expt":4}
        fidelities_dir = "train"
        subsample_dict = {"pbe": 1, "scan": 1, "gllb-sc": 1, "hse": 1, "expt": 1}
    else:
        fidelities_dir = "train_homemade"
        fidelity_map = {
            "GGA": 0,
            "SCAN": 1,
            "GLLBSC": 2,
            "HSE": 3,
            "EXPT": 4
            }
        subsample_dict = {"GGA": 0.6, "SCAN": 1, "GLLBSC": 1, "HSE": 1, "EXPT": 1}
    #fidelities_dir = "train"
    trainer = MultiTrainer(model_params=model_params, subsample_dict=subsample_dict, training_params=training_params, fidelity_map=fidelity_map, property_name='BG', fidelities_dir=fidelities_dir)
    print(f"\nTrial {trial.number}: Parameters: {trial.params}")
    combined_train_df, test_datasets= trainer.prepare_datasets_for_multifidelity()
    try:
        model, preprocess, mean, std = trainer.train_multifidelity_model(combined_train_df, trial=trial)
    except optuna.TrialPruned:
        raise
    except Exception as e:
        print(f"An error occurred during training for trial {trial.number}: {e}")
        return float('inf') # Indicate failure
    nmaes = {}
    total_nmaes = 0
    for dataset_name, dataset in test_datasets.items():
        test_mean, test_std = dataset[trainer.property_name].mean(), dataset[trainer.property_name].std()
        test_loader = trainer.create_test_dataloader(dataset, preprocess, mean, std)
        metrics, predictions, targets = trainer.evaluate_on_fidelity( model,test_loader,  mean, std)
        nmae = metrics["mae"]/test_std
        nmaes[dataset_name] = {'nmae': nmae}
        print(f"Dataset: {dataset_name}, NMAE: {nmae},")
        total_nmaes += nmae
    return total_nmaes
    
study_name = "latest_multifidelity_bandgap_study" # You can change this
storage_name = f"sqlite:///{study_name}.db"
num_parallel_jobs = 6
study = optuna.create_study(
    study_name=study_name,
    storage=storage_name, # Specify storage for persistence and dashboard
    load_if_exists=True, # Load an existing study with the same name if it exists
    direction="minimize",
    pruner=optuna.pruners.MedianPruner(n_startup_trials=3, n_warmup_steps=5, interval_steps=1,
    )
)

print(f"Using Optuna study: {study_name}")
print(f"Storage: {storage_name}")
print(f"Number of trials in study before optimization: {len(study.trials)}")


try:
    # Adjust n_trials as needed for your hyperparameter search
    study.optimize(objective, n_trials=80, n_jobs = num_parallel_jobs) # Example: run for 20 new trials
except ImportError as e:
    print(f"Could not run Optuna study due to import error: {e}")
except RuntimeError as e: # Catch potential runtime errors, e.g. DB issues
    print(f"A runtime error occurred during the Optuna study: {e}")
except Exception as e:
    print(f"An unexpected error occurred during the Optuna study: {e}")


print("\n--- Optuna Study Finished ---")
print(f"Study statistics for '{study_name}':")
print(f"  Number of finished trials (total in DB): {len(study.trials)}")

try:
    # Filter for completed trials in this session if needed, or just show overall best
    completed_trials = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
    if completed_trials:
        print(f"  Number of successfully completed trials in DB: {len(completed_trials)}")
        print("\nBest trial overall (from storage):")
        best_trial = study.best_trial
        print(f"  Value (Sum of NMAEs): {best_trial.value:.4f}")
        print("  Params: ")
        for key, value in best_trial.params.items():
            print(f"    {key}: {value}")
    else:
        print("No trials were completed successfully in this study.")

except ValueError: # Optuna raises ValueError if no trials are completed
    print("No trials were completed successfully in the study, or best_trial is not available.")
except Exception as e:
    print(f"Error retrieving best trial: {e}")

print(f"\nTo visualize the study, run the Optuna dashboard (if installed):")
print(f"optuna-dashboard {storage_name}")