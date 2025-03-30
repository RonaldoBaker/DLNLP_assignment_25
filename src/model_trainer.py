import os
import sys
import time
import matplotlib.pyplot as plt
import torch
from torchtext.data.metrics import bleu_score
from torch.nn.utils import clip_grad_norm_
from torch.nn.functional import log_softmax
from nltk.translate.bleu_score import corpus_bleu
from tqdm import tqdm
import heapq

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

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

                early_stopping(self.val_losses[-1])
                if early_stopping.early_stop:
                    print("Early stopping")
                    break

            print(f"Epoch {epoch + 1} | Train Loss: {self.train_losses[-1]} | Val Loss: {self.val_losses[-1]}")
            print()

        # Mark the end time
        end_time = time.time()

        elapsed_time = time.strftime("Hh %Mm %Ss", time.gmtime(end_time - start_time))
        print(f"Training completed in {elapsed_time}")


    def evaluate_bleu(self, tgt_vocab, max_len, type: str = "greedy", beam_width: int = None) -> float:
        """
        Evaluate the model on the test set using BLEU score.
        
        Arg(s):
            - tgt_vocab: Target vocabulary mapping tokens to indices; should contain "<sos>" and "<eos>".
            - max_len: Maximum length for generated translations.
            - type: Decoding type, either "greedy" or "beam".
            - beam_width: Width of the beam for beam search decoding (only used if type is "beam").
        
        Returns:
            - float: The corpus BLEU score.
        """
        # Get the index-to-word mapping from the torchtext Vocab object.
        # Vocab.itos is a list where index corresponds to the token.
        itos = tgt_vocab.get_itos()

        # Get start and end token indices using the stoi attribute.
        tgt_start_token = tgt_vocab["<sos>"]
        tgt_end_token = tgt_vocab["<eos>"]

        if tgt_start_token is None or tgt_end_token is None:
            raise ValueError("Target vocabulary must contain <sos> and <eos> tokens.")

        candidates = []
        references = []
        self.model.eval()

        with torch.no_grad():
            # Iterate over the test set batches
            for src, tgt in tqdm(self.test_loader, desc="Testing Transformer", leave=True, unit="batch"):
                batch_size = src.size(0)

                # If using greedy decoding.
                if type == "greedy":
                    # Initialize predictions with the start token repeated for the batch.
                    preds = torch.full((batch_size, 1), tgt_start_token, dtype=torch.long, device=self.device)

                    # Auto-regressive decoding loop (using greedy decoding).
                    for _ in range(max_len - 1):
                        outputs = self.model(src, preds)  # shape: (batch_size, seq_len, vocab_size)
                        next_token = outputs[:, -1, :].argmax(dim=-1, keepdim=True)
                        preds = torch.cat([preds, next_token], dim=1)
                        # If every sequence in the batch has generated the <eos> token, break early.
                        if (next_token == tgt_end_token).all():
                            break

                    # Process each sequence in the batch.
                    for pred_seq, tgt_seq in zip(preds, tgt):
                        # Convert prediction token IDs to words, skipping the <sos> token.
                        pred_tokens = []
                        for token_id in pred_seq[1:]:
                            if token_id.item() == tgt_end_token:
                                break
                            # Make sure the index is within bounds.
                            token = itos[token_id.item()] if token_id.item() < len(itos) else "<unk>"
                            pred_tokens.append(token)
                        candidates.append(pred_tokens)

                        # Process the reference translation (skipping the <sos> token).
                        ref_tokens = []
                        for token_id in tgt_seq[1:]:
                            if token_id.item() == tgt_end_token:
                                break
                            token = itos[token_id.item()] if token_id.item() < len(itos) else "<unk>"
                            ref_tokens.append(token)
                        # corpus_bleu expects a list of reference sentences per candidate.
                        references.append([ref_tokens])
                
                # If using beam search decoding.
                elif type == "beam":
                    # Initialize beams for each batch element
                    beams = [[(0.0, [tgt_start_token])] for _ in range(batch_size)]
                    completed_sequences = [[] for _ in range(batch_size)]
                    
                    for _ in range(max_len - 1):
                        for i, beam in enumerate(beams):
                            new_beam = []
                            for score, seq in beam:
                                if seq[-1] == tgt_end_token:
                                    completed_sequences[i].append((score, seq))
                                    continue
                                
                                # Expand the sequence
                                seq_tensor = torch.tensor(seq, dtype=torch.long, device=self.device).unsqueeze(0)
                                src_i = src[i].unsqueeze(0)
                                outputs = self.model(src_i, seq_tensor)  # shape: (1, seq_len, vocab_size)
                                logits = outputs[:, -1, :]
                                log_probs = log_softmax(logits, dim=-1)
                                
                                # Get top beam_width candidates
                                top_log_probs, top_indices = torch.topk(log_probs, beam_width, dim=-1)
                                
                                for log_prob, token_idx in zip(top_log_probs.squeeze(0), top_indices.squeeze(0)):
                                    new_seq = seq + [token_idx.item()]
                                    new_score = score + log_prob.item()
                                    heapq.heappush(new_beam, (new_score, new_seq))
                                    
                            # Keep only the best beam_width candidates
                            beams[i] = heapq.nlargest(beam_width, new_beam)
                            
                        # If all beams are complete, break early
                        if all(len(c) > 0 for c in completed_sequences):
                            break
                    
                    # Select best sequences
                    for i in range(batch_size):
                        best_seq = max(completed_sequences[i] or beams[i], key=lambda x: x[0])[1]
                        pred_tokens = [itos[tok] for tok in best_seq[1:] if tok != tgt_end_token]
                        candidates.append(pred_tokens)
                        
                        # Process references
                        ref_tokens = [itos[tok.item()] for tok in tgt[i, 1:] if tok.item() != tgt_end_token]
                        references.append([ref_tokens])

                else:
                    raise ValueError("Invalid decoding type. Use 'greedy' or 'beam'.")

        bleu = corpus_bleu(references, candidates)
        print(f"BLEU score: {bleu:.4f}")
        return bleu


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
