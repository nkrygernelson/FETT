

from multi_main import MultiTrainer 
pooling_params = {"hierarchical":{"num_motifs":6}}
model_params = {
    "num_elements": 118,
    "num_fidelities": 5,
    "embedding_dim": 236,
    "fidelity_dim": 16,
    "num_blocks": 4,
    "num_heads": 4,
    "hidden_dim": 160,
    "dropout": 0.10614,
    "pooling_type": "weighted",
    "pooling_params":pooling_params,
}

training_params = {
    "multi_train_split": 0.8,
    "epochs": 90,
    "batch_size": 64,
    "learning_rate": 0.0001147759,
    "weight_decay": 0.000001141,
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
        "SCAN": 0,
        "GLLBSC": 0,
        "HSE": 0,
        "EXPT": 0
        }
    subsample_dict = {"GGA": 0.5, "SCAN": 1, "GLLBSC": 1, "HSE": 1, "EXPT": 1}

trainer = MultiTrainer(model_params=model_params, subsample_dict=subsample_dict,
                       training_params=training_params, fidelities_dir=fidelities_dir, 
                       fidelity_map=fidelity_map, optunize=True)
results, plot_paths = trainer.run_multifidelity_experiments()
print(results)
