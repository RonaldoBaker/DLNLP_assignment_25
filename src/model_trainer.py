import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import matplotlib.pyplot as plt
import torch
from torch.nn.utils import clip_grad_norm_
from tqdm import tqdm
from src.early_stopping import EarlyStopping

class TransformerTrainer():
    def __init__(self, train_loader, val_loader, test_loader, epochs, optimiser, scheduler, loss_func, model, device):
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.epochs = epochs
        self.optimiser = optimiser
        self.scheduler = scheduler
        self.loss_func = loss_func
        self.model = model
        self.device = device
        # Store the losses while training to plot loss curves
        self.train_losses = []
        self.val_losses = []

    def train(self, patience: int = 5, delta: int = 0):
        # Define early stopping
        early_stopping = EarlyStopping(patience=patience, delta=delta)

        # Training loop
        for epoch in range(self.epochs):
            self.model.train()
            train_loss = 0

            for src, tgt in tqdm(self.train_loader, desc="Training Transformer", leave=True):
                # Move the source and target tensors to the device
                src = src.to(self.device)
                tgt = tgt.to(self.device)

                self.optimiser.zero_grad()
                output = self.model(src, tgt[:, :-1])
                
                output = output.reshape(-1, output.shape[-1])
                tgt = tgt[:, 1:].reshape(-1)

                loss = self.loss_func(output, tgt)
                loss.backward()

                # Clip the gradients to avoid exploding gradients
                clip_grad_norm_(self.model.parameters(), max_norm=1)
                self.optimiser.step()

                train_loss += loss.item()

            train_loss /= len(self.train_loader)
            self.train_losses.append(train_loss)

            
            with torch.no_grad():
                self.model.eval()
                val_loss = 0

                for src, tgt in tqdm(self.val_loader, desc="Validating Transformer", leave=True):
                    src = src.to(self.device)
                    tgt = tgt.to(self.device)

                    output = self.model(src, tgt[:, :-1])
                    output = output.reshape(-1, output.shape[-1])
                    tgt = tgt[:, 1:].reshape(-1)

                    loss = self.loss_func(output, tgt)
                    val_loss += loss.item()
                
                val_loss /= len(self.val_loader)
                self.scheduler.step(train_loss)
                self.val_losses.append(val_loss)

                early_stopping(self.val_losses[-1])
                if early_stopping.early_stop:
                    print("Early stopping")
                    break

            print(f"Epoch {epoch + 1} | Train Loss: {self.train_losses[-1]} | Val Loss: {self.val_losses[-1]}")

    def plot_loss_curves(self, epoch_resolution: int, path: str):
        sampled_epochs = list(range(0, len(self.train_losses), epoch_resolution))
        sampled_train_losses = self.train_losses[::epoch_resolution]
        sampled_val_losses = self.val_losses[::epoch_resolution]

        plt.plot(sampled_epochs, sampled_train_losses, label="Training Loss")
        plt.plot(sampled_epochs, sampled_val_losses, label="Validation Loss")
        plt.grid()
        plt.title("Training and Validation Loss Curves")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.legend()
        if path:
            plt.savefig(path)
        plt.show()
