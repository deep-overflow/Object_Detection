import torch
import torch.nn as nn
from utils import get_optimizer, get_lr_scheduler

from tqdm import tqdm
import wandb

class Trainer:
    def __init__(self, device, model, dataloader, configs):
        self.device = device
        self.model = model
        self.dataloader = dataloader
        self.configs = configs

        self.optimizer = get_optimizer(configs.optimizer, self.model.parameters())
        self.lr_scheduler = get_lr_scheduler(configs.lr_scheduler, self.optimizer)

    def run(self):
        self.model.load_state_dict(torch.load('weight1.pth'))

        criterion = nn.CrossEntropyLoss()

        self.model = self.model.to(self.device)
        
        for epoch in range(self.configs.epochs):
            print(f"Epoch : {epoch + 1} ==========")

            # Train
            self.model.train()
            train_loss = 0.0
            n_samples = len(self.dataloader["train"].dataset)
            for inputs, labels in tqdm(self.dataloader["train"]):
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                labels = labels.reshape(-1, 256, 256).type(torch.int)
                n_samples_batch = labels.shape[0]

                outputs = self.model(inputs)

                loss = criterion(outputs, labels)

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                self.lr_scheduler.step()

                train_loss += loss.item() * n_samples_batch
            
            train_loss = train_loss / n_samples
            print(f"Train Loss : {train_loss}")

            # Eval
            self.model.eval()
            eval_loss = 0.0
            n_samples = len(self.dataloader["eval"].dataset)
            for inputs, labels in tqdm(self.dataloader["eval"]):
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                labels = labels.reshape(-1, 256, 256).type(torch.int)
                n_samples_batch = labels.shape[0]
                
                with torch.no_grad():
                    outputs = self.model(inputs)
                    
                    loss = criterion(outputs, labels)
                
                eval_loss += loss.item() * n_samples_batch

            eval_loss = eval_loss / n_samples
            print(f"Eval Loss : {eval_loss}")
        
        torch.save(self.model.state_dict(), "weight2.pth")
