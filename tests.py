
from multi_main import MultiTrainer 
import os

load_path = os.path.join("data","runs","expanded")

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

fidelity_map = {"gga":0, "gga+u":1, "pbe_sol":2,"scan":3, "gllbsc":4, "hse":5,"expt":6}
subsample_dict = {"gga":1, "gga+u":1, "pbe_sol":1,"scan":1, "gllbsc":1, "hse":1,"expt":1}

trained_model_path = None
trainer = MultiTrainer(model_params=model_params, subsample_dict=subsample_dict,
                       training_params=training_params,model_params_path=None,
                       property_name="bg",
                       trained_model_path=None, 
                       fidelity_map=fidelity_map, optunize=False,collab=True)
results, plot_paths = trainer.run_multifidelity_experiments()
print(results)