
from multi_main import MultiTrainer 
import os

load_path = os.path.join("data","runs","standard")

pooling_params = {} # num_motifs is empty in the table
model_params = {
    "embedding_dim": 172,
    "fidelity_dim": 20,
    "hidden_dim": 288,
    "num_blocks": 5,
    "num_fidelities":5,
    "num_elements":118,
    "num_heads": 16,
    "dropout": 0.17912079167770537,
    "pooling_type": "weighted",
    "pooling_params": pooling_params,
}


training_params = {
    "epochs": 110,
    "batch_size": 64,
    "learning_rate": 0.000351208662205836,
    "weight_decay": 0.000001093212600,
    "load_data": True,
    "load_path": load_path
}

fidelity_map = {"pbe":0, "scan":1, "gllb-sc":2, "hse":3, "expt":4}
subsample_dict = {"pbe": 1, "scan": 1, "gllb-sc": 1, "hse": 1, "expt": 1}

trained_model_path = None
trainer = MultiTrainer(model_params=model_params, subsample_dict=subsample_dict,
                       training_params=training_params,model_params_path=None,
                       property_name="BG",
                       trained_model_path=None, 
                       fidelity_map=fidelity_map, optunize=False,collab=True)
results, plot_paths = trainer.run_multifidelity_experiments()
print(results)