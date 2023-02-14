import torch
import torch.nn as nn
import torch.optim as optim
from models import get_model
from dataloaders import get_dataloader
from trainer import Trainer

import wandb
import argparse
from omegaconf import OmegaConf

if __name__ == "__main__":
    # Args
    parser = argparse.ArgumentParser(description="Arguments for Segmentation Models")
    parser.add_argument("--conf_file", type=str, default="basic_train.yaml")
    args = parser.parse_args()

    # Configs
    configs = OmegaConf.load("configs/" + args.conf_file)

    # Wandb
    # wandb.init(
    #     project="Segmentation",
    #     config=configs,
    # )

    # Device
    device = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
    device = torch.device("cuda") if torch.backends.cuda.is_available() else torch.device("cpu")
    
    print(f"Device : {device}")

    # DataLoader
    dataloader = get_dataloader(configs.dataset)

    # Model
    model = get_model(configs.model)

    # Trainer
    trainer = Trainer(device, model, dataloader, configs.train)

    # Train
    trainer.run()