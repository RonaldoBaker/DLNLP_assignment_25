import os
import sys
import time
import matplotlib.pyplot as plt
import torch
from torch.nn.utils import clip_grad_norm_
from tqdm import tqdm

# Append project root to sys.path
project_root = os.path.join(os.path.dirname(__file__), "..")
if project_root not in sys.path:
    sys.path.append(project_root)

from src.early_stopping import EarlyStopping
from utils.config import config

class TransformerTrainer:
    def __init__(self, train_loader, val_loader, epochs, optimiser, scheduler, loss_func, model, device):
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.epochs = epochs
        self.optimiser = optimiser
        self.scheduler = scheduler
        self.loss_func = loss_func
        self.model = model
        self.device = device
        # Store the losses while training to plot loss curves
        self.train_losses = []
        self.val_losses = []
        self.best_val_loss = float("inf")


    def save_checkpoint(self, mode: str):
        """
        Saves the model checkpoint.

        Arg(s):
            - mode (str): The mode of the checkpoint, either "best" or "last".
            - path (str): The path to save the checkpoint. If empty, saves in the current directory.
        """
        path = config.SAVE_FILEPATH + f"checkpoints/{mode}_model.pth"

        # Define checkpoint
        checkpoint = {
            "model": self.model.state_dict()
        }
        torch.save(checkpoint, path)


    def train(self, patience: int = 5, delta: int = 0):
        # Define early stopping
        early_stopping = EarlyStopping(patience=patience, delta=delta)

        # Start timer
        start_time = time.time()
        # Training loop
        for epoch in range(self.epochs):
            self.model.train()
            train_loss = 0

            for src, tgt in tqdm(self.train_loader, desc="Training Transformer", leave=True, unit="batch"):

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

                for src, tgt in tqdm(self.val_loader, desc="Validating Transformer", leave=True, unit="batch"):

                    output = self.model(src, tgt[:, :-1])
                    output = output.reshape(-1, output.shape[-1])
                    tgt = tgt[:, 1:].reshape(-1)

                    loss = self.loss_func(output, tgt)
                    val_loss += loss.item()

            val_loss /= len(self.val_loader)
            self.scheduler.step(val_loss)
            self.val_losses.append(val_loss)

            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.save_checkpoint("best")
                print(f"Best model saved at epoch {epoch + 1}")

            early_stopping(self.val_losses[-1])
            if early_stopping.early_stop:
                print("Early stopping")
                break

            print(f"Epoch {epoch + 1} | Train Loss: {self.train_losses[-1]} | Val Loss: {self.val_losses[-1]}")
            print()

        # Save the last model
        self.save_checkpoint(mode="last")

        # Mark the end time
        end_time = time.time()

        elapsed_time = time.strftime("Hh %Mm %Ss", time.gmtime(end_time - start_time))
        print(f"Training completed in {elapsed_time}")


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
