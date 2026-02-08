from fett.model import Model
from fett.data import MyDataset
from omegaconf import OmegaConf
import hydra
import torch
@hydra.main(config_path="../../configshydra", config_name="config", version_base="1.3")
def main(cfg:DictConfig):
    dataset = MyDataset("data/raw")
    model = Model()

    # add rest of your training code here

if __name__ == "__main__":
    train()
