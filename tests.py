
from multi_main import MultiTrainer 
import os

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
load_path = os.path.join("data","runs","only_new_on_expt")
training_params = {
    "multi_train_split": 0.8,
    "epochs": 90,
    "batch_size": 64,
    "learning_rate": 0.0001147759,
    "weight_decay": 0.000001141,
    "load_data":True,
    "load_path":load_path
}


fidelity_map = {"pbe":0, "scan":1, "gllb-sc":2, "hse":3, "expt":4}
fidelities_dir = "train"
subsample_dict = {"pbe": 1, "scan": 1, "gllb-sc": 1, "hse": 1, "expt": 1}

model_params_path = os.path.join("runs", "run_2025-09-18_093927_b25658f2", "model_params.json")
trained_model_path = os.path.join("runs", "run_2025-09-18_093927_b25658f2", "best_model_run_2025-09-18_093927_b25658f2.pt")

trainer = MultiTrainer(model_params=model_params, subsample_dict=subsample_dict,
                       training_params=training_params,model_params_path=model_params_path,
                       trained_model_path=trained_model_path, 
                       fidelity_map=fidelity_map, optunize=False,collab=True)
results, plot_paths = trainer.run_multifidelity_experiments()
print(results)