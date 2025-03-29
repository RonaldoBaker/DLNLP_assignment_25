import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import matplotlib.pyplot as plt
import torch
from torchtext.data.metrics import bleu_score
from nltk.translate.bleu_score import corpus_bleu
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

            for src, tgt in tqdm(self.train_loader, desc="Training Transformer", leave=True, unit="batch"):
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

                for src, tgt in tqdm(self.val_loader, desc="Validating Transformer", leave=True, unit="batch"):
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
            print()


    def evaluate_bleu_greedy(self, tgt_vocab, src_pad_index, max_len):
        """
        Evaluate the model on the test set using BLEU score.
        
        Args:
            test_loader: DataLoader yielding tuples (src, tgt) of shape (batch_size, seq_len).
            model: The transformer model.
            device: torch.device.
            tgt_vocab: Target vocabulary mapping tokens to indices; should contain "<sos>" and "<eos>".
            src_pad_index: Padding index for the source.
            max_len: Maximum length for generated translations.
        
        Returns:
            bleu (float): The corpus BLEU score.
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
            for src, tgt in tqdm(self.test_loader, desc="Testing Transformer", leave=True, unit="batch"):
                src = src.to(self.device)
                batch_size = src.size(0)
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

        bleu = corpus_bleu(references, candidates)
        print(f"BLEU score: {bleu:.4f}")
        return bleu


    def evaluate_bleu_beam(self, tgt_vocab, src_pad_index, max_len, beam_width=3):
        """
        Evaluate the model on the test set using BLEU score with beam search decoding.
        
        Args:
            tgt_vocab: torchtext Vocab object for the target language. Must contain "<sos>" and "<eos>" tokens.
            src_pad_index: Padding index for the source.
            max_len: Maximum length for generated translations.
            beam_width: Beam width.
        
        Returns:
            bleu (float): The corpus BLEU score.
        """
        # Get the index-to-word mapping from the torchtext Vocab object.
        itos = tgt_vocab.get_itos()

        # Retrieve start and end token indices.
        tgt_start_token = tgt_vocab["<sos>"]
        tgt_end_token = tgt_vocab["<eos>"]

        if tgt_start_token is None or tgt_end_token is None:
            raise ValueError("Target vocabulary must contain <sos> and <eos> tokens.")

        candidates = []
        references = []
        self.model.eval()

        with torch.no_grad():
            # Iterate over the test set batches.
            for src, tgt in tqdm(self.test_loader, desc="Testing Transformer with Beam Search", leave=True):
                src = src.to(self.device)
                batch_size = src.size(0)
                # Process each sample in the batch individually.
                for i in range(batch_size):
                    src_i = src[i].unsqueeze(0)  # shape: (1, sequence_length)
                    # Initialize beam with the start token and zero score.
                    beam = [([tgt_start_token], 0.0)]
                    finished_candidates = []
                    # Beam search decoding loop.
                    for _ in range(max_len - 1):
                        new_beam = []
                        for seq, score in beam:
                            # If the sequence is already finished, keep it.
                            if seq[-1] == tgt_end_token:
                                finished_candidates.append((seq, score))
                                continue
                            # Convert the current sequence to a tensor.
                            seq_tensor = torch.tensor(seq, dtype=torch.long, device=self.device).unsqueeze(0)
                            outputs = self.model(src_i, seq_tensor)  # shape: (1, seq_len, vocab_size)
                            logits = outputs[:, -1, :]  # get logits at the last time step.
                            # Compute log probabilities.
                            log_probs = torch.log_softmax(logits, dim=-1)  # shape: (1, vocab_size)

                            # Get the top beam_width tokens and their log probabilities.
                            top_log_probs, top_indices = torch.topk(log_probs, beam_width, dim=-1)
                            top_log_probs = top_log_probs.squeeze(0)
                            top_indices = top_indices.squeeze(0)
        
                            for log_prob, token_idx in zip(top_log_probs, top_indices):
                                new_seq = seq + [token_idx.item()]
                                new_score = score + log_prob.item()
                                new_beam.append((new_seq, new_score))
                        # If no candidates were expanded, break.
                        if not new_beam:
                            break
                        # Sort candidates by score (highest first) and prune.
                        new_beam = sorted(new_beam, key=lambda x: x[1], reverse=True)[:beam_width]
                        beam = new_beam
                        # If every candidate in the beam has completed the sequence, end early.
                        if all(seq[-1] == tgt_end_token for seq, _ in beam):
                            finished_candidates.extend(beam)
                            break
                    # Choose the best sequence among finished candidates if available.
                    if finished_candidates:
                        best_seq = max(finished_candidates, key=lambda x: x[1])[0]
                    else:
                        best_seq = beam[0][0]

                    # Convert token ids (skipping <sos>) into words.
                    pred_tokens = []
                    for token_id in best_seq[1:]:
                        if token_id == tgt_end_token:
                            break
                        token = itos[token_id] if token_id < len(itos) else "<unk>"
                        pred_tokens.append(token)
                    candidates.append(pred_tokens)

                # Process references for this batch.
                for tgt_seq in tgt:
                    ref_tokens = []
                    for token_id in tgt_seq[1:]:
                        if token_id.item() == tgt_end_token:
                            break
                        token = itos[token_id.item()] if token_id.item() < len(itos) else "<unk>"
                        ref_tokens.append(token)
                    # corpus_bleu expects a list of reference sentences per candidate.
                    references.append([ref_tokens])

        bleu = corpus_bleu(references, candidates)
        print(f"BLEU score (Beam Search): {bleu:.4f}")
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
