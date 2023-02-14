import torch
import torch.nn as nn
from models import get_model
from dataloaders import get_dataloader
from utils import mask2image, save_tensor_img
import matplotlib.pyplot as plt
from PIL import Image

import wandb
import argparse
from omegaconf import OmegaConf

if __name__ == "__main__":
    # Args
    parser = argparse.ArgumentParser(description="Arguments for Segmentation Models")
    parser.add_argument("--conf_file", type=str, default="basic_eval.yaml")
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
    # device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    
    print(f"Device : {device}")

    # DataLoader
    dataloader = get_dataloader(configs.dataset)

    # Model
    model = get_model(configs.model)
    model.load_state_dict(torch.load("weight2.pth"))
    model.eval()
    model = model.to(device)

    # Softmax
    softmax = nn.Softmax2d()

    # Train
    n_samples_train = len(dataloader["train"].dataset)
    for idx in range(configs.n_samples.train):
        image, label = dataloader["train"].dataset[idx]

        save_tensor_img(image, f"results/image_{idx}.jpeg")
        label = label.type(torch.long)
        label = mask2image(label)
        save_tensor_img(label.type(torch.uint8), f"results/label_{idx}.jpeg")

        image = image.reshape(1, 3, 256, 256).to(device)

        with torch.no_grad():
            output = model(image)
        prob = softmax(output)
        pred = torch.argmax(prob, dim=1)

        mask = mask2image(pred.cpu())
        save_tensor_img(mask.type(torch.uint8), f"results/mask_{idx}.jpeg")




    # for idx in range(configs.n_samples.eval):
    #     image, label = dataloader["eval"].dataset[idx]
    #     image = image.reshape(1, 3, 256, 256)
    #     with torch.no_grad():
    #         output = model(image)
    #     output = output.reshape(3, 256, 256)

