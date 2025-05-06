import os
import sys
import time
import matplotlib.pyplot as plt
import torch
from torch.nn.utils import clip_grad_norm_
from tqdm import tqdm
from nltk.translate.bleu_score import corpus_bleu, SmoothingFunction

# Append project root to sys.path
project_root = os.path.join(os.path.dirname(__file__), "..")
if project_root not in sys.path:
    sys.path.append(project_root)

from src.early_stopping import EarlyStopping

class TransformerTrainer:
    def __init__(self, train_loader, val_loader, tgt_vocab, max_len, epochs, optimiser, scheduler, loss_func, model, device, log_dir):
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.tgt_vocab_itos = tgt_vocab.get_itos()
        self.tgt_start_token = tgt_vocab["<sos>"]
        self.tgt_end_token = tgt_vocab["<eos>"]
        self.max_len = max_len
        self.epochs = epochs
        self.optimiser = optimiser
        self.scheduler = scheduler
        self.loss_func = loss_func
        self.model = model
        self.device = device
        # Store the losses while training to plot loss curves
        self.train_losses = []
        self.val_losses = []
        self.val_bleus = []
        self.best_val_loss = float("inf")
        self.best_val_bleu = float("-inf")
        self.smoothie = SmoothingFunction().method4 # Use method4 for BLEU score smoothing
        self.log_dir = log_dir


    def save_checkpoint(self, mode: str):
        """
        Saves the model checkpoint.

        Arg(s):
            - mode (str): The mode of the checkpoint, either "best" or "last".
            - path (str): The path to save the checkpoint. If empty, saves in the current directory.
        """
        path = os.path.join(self.log_dir, f"{mode}_model.pth")

        # Define checkpoint
        checkpoint = {
            "model": self.model.state_dict()
        }

        torch.save(checkpoint, path)


    def calculate_validation_bleu_score(self, src, tgt):
        # Get batch size
        batch_size = tgt.size(0)

        # Initialize lists to store predictions and references
        candidates = []
        references = []

        # Initialize predictions with the start token repeated for the batch.
        preds = torch.full((batch_size, 1), self.tgt_start_token, dtype=torch.long, device=self.device)

        # Auto-regressive decoding loop (using greedy decoding).
        for _ in range(self.max_len - 1):
            outputs = self.model(src, preds)  # shape: (batch_size, seq_len, vocab_size)
            next_token = outputs[:, -1, :].argmax(dim=-1, keepdim=True)
            preds = torch.cat([preds, next_token], dim=1)
            # If every sequence in the batch has generated the <eos> token, break early.
            if (next_token == self.tgt_end_token).all():
                break

        # Process each sequence in the batch.
        for pred_seq, tgt_seq in zip(preds, tgt):
            # Convert prediction token IDs to words, skipping the <sos> token.
            pred_tokens = []
            for token_id in pred_seq[1:]:
                if token_id.item() == self.tgt_end_token:
                    break
                # Make sure the index is within bounds.
                token = self.tgt_vocab_itos[token_id.item()] if token_id.item() < len(self.tgt_vocab_itos) else "<unk>"
                pred_tokens.append(token)
            # corpus bleu expects a list of reference sentences per candidate (list of lists)
            candidates.append(pred_tokens)

            # Process the reference translation (skipping the <sos> token).
            ref_tokens = []
            for token_id in tgt_seq[1:]:
                if token_id.item() == self.tgt_end_token:
                    break
                token = self.tgt_vocab_itos[token_id.item()] if token_id.item() < len(self.tgt_vocab_itos) else "<unk>"
                ref_tokens.append(token)
            # corpus_bleu expects a list of reference sentences per candidate (list of lists)
            references.append([ref_tokens])

        # Calculate BLEU score using corpus_bleu from nltk
        bleu_score = corpus_bleu(references, candidates, smoothing_function=self.smoothie)
        return bleu_score


    def train(self, patience: int = 5, delta: int = 0):
        # Define early stopping
        early_stopping = EarlyStopping(patience=patience, delta=delta, mode="max")

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
                val_bleu = 0

                for src, tgt in tqdm(self.val_loader, desc="Validating Transformer", leave=True, unit="batch"):

                    # Calculate BLEU score first before reshaping
                    bleu_score = self.calculate_validation_bleu_score(src, tgt)
                    val_bleu += bleu_score

                    # Forward pass
                    output = self.model(src, tgt[:, :-1])
                    output = output.reshape(-1, output.shape[-1])
                    tgt = tgt[:, 1:].reshape(-1)

                    # Calculate loss
                    loss = self.loss_func(output, tgt)
                    val_loss += loss.item()

            val_loss /= len(self.val_loader)
            val_bleu /= len(self.val_loader)
            self.scheduler.step(val_loss)
            self.val_losses.append(val_loss)
            self.val_bleus.append(val_bleu)

            if val_bleu > self.best_val_bleu:
                self.best_val_bleu = val_bleu
                self.save_checkpoint("best")
                print(f"Best model saved at epoch {epoch + 1}")

            early_stopping(self.val_bleus[-1])
            if early_stopping.early_stop:
                print("Early stopping")
                break

            print(f"Epoch {epoch + 1} | Train Loss: {self.train_losses[-1]} | Val Loss: {self.val_losses[-1]} | Val BLEU: {self.val_bleus[-1]}")
            print()

        # Save the last model
        self.save_checkpoint(mode="last")

        # Mark the end time
        end_time = time.time()

        elapsed_seconds = end_time - start_time
        hours, remainder = divmod(elapsed_seconds, 3600)
        minutes, seconds = divmod(remainder, 60)
        self.elapsed_time = f"{int(hours):02}:{int(minutes):02}:{int(seconds):02}"
        print(f"Training completed in {self.elapsed_time}")


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
