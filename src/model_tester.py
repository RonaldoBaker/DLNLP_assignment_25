import os
import sys
import numpy as np
import torch
from torch.nn.functional import log_softmax
from nltk.translate.bleu_score import corpus_bleu
from tqdm import tqdm
import heapq
from evaluate import load
from rich.table import Table
from rich.console import Console

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.config import config

class TransformerTester:
    def __init__(self, test_loader, model, device):
        self.test_loader = test_loader
        self.model = model
        self.device = device
        self.bertscore = load("bertscore")

    
    def load_model(self):
        path = config.SAVE_FILEPATH + f"checkpoints/best_model.pth" 

        if not os.path.exists(path):
            raise FileNotFoundError(f"Checkpoint file not found at {path}")
        else:
            checkpoint = torch.load(path, map_location=self.device, weights_only=True)
            self.model.load_state_dict(checkpoint["model"])


    def calculate_bertscore(self, references: list[list[str]], candidates: list[list[str]]):
        """
        Calculate BERTScore for the given references and candidates.

        Arg(s):
            - references: List of reference translations (list of lists).
            - candidates: List of candidate translations (list of lists).

        Returns:
            - tuple: Mean precision, recall, and F1 score.
        """
        # Convert list of lists to list of strings
        references = [" ".join(ref[0]) for ref in references]
        candidates = [" ".join(cand) for cand in candidates]

        results = self.bertscore.compute(predictions=candidates,
                                         references=references,
                                         lang="es",
                                         model_type="bert-base-uncased",
                                         device=self.device)
        precision = np.mean(np.array(results["precision"]))
        recall = np.mean(np.array(results["recall"]))
        f1 = np.mean(np.array(results["f1"]))

        return precision, recall, f1


    def calculate_oov_rates(self, sequences: dict):
        """
        Calculate the Out-Of-Vocabulary (OOV) rates for the given potential vocabularies.
        """
        vocabs = {}
        for name, seq in sequences.items():
            vocabs[name] = self.get_vocab(seq)

        train_vocab = vocabs["train_set_vocab"]
        test_vocab = vocabs["test_set_vocab"]
        output_model_vocab = vocabs["output_model_vocab"]

        # Calculate the reference OOV rate
        reference_oov_rate = len(test_vocab - train_vocab) / len(test_vocab) * 100

        # Calculate the prediction OOV rate
        prediction_oov_rate = len(output_model_vocab - train_vocab) / len(output_model_vocab) * 100

        return reference_oov_rate, prediction_oov_rate


    def calculate_unknown_rate(self, sequences: list[list[str]]):
        """
        Calculate the unknown rate for the given list of lists.
        Arg(s):
            - list[list[str]]: List of lists of strings.
        Returns:
            - float: The unknown rate.
        """
        total_tokens = sum(len(sublist) for sublist in sequences)
        unknown_tokens = sum(1 for sublist in sequences for token in sublist if token == "<unk>")
        return unknown_tokens / total_tokens * 100


    def get_vocab(self, dataset: list):
        """
        Get the vocabulary from the dataset
        Arg(s):
        - dataset: The dataset to get the vocabulary from - it should be a TokenDataset object
        """
        vocab = set()
        if isinstance(dataset, list):

            if all(isinstance(item, list) for item in dataset):
                # If the dataset is a list of lists, iterate over each sentence
                for sentence in dataset:
                    vocab.update(sentence)

            elif all(isinstance(item, tuple) for item in dataset):
                for _, tgt_tensor in dataset:
                    tgt_tokens = tgt_tensor.cpu().numpy().tolist()
                    vocab.update(tgt_tokens)

        return vocab


    def evaluate(self, tgt_vocab, max_len, train_set, test_set, type: str = "greedy", beam_width: int = None):
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
        # Load the best model
        self.load_model()

        # Get the index-to-word mapping from the torchtext Vocab object.
        # Vocab.itos is a list where index corresponds to the token.
        itos = tgt_vocab.get_itos()

        # Get start and end token indices using the stoi attribute.
        tgt_start_token = tgt_vocab["<sos>"]
        tgt_end_token = tgt_vocab["<eos>"]

        if tgt_start_token is None or tgt_end_token is None:
            raise ValueError("Target vocabulary must contain <sos> and <eos> tokens.")

        candidates = []
        candidates_ids = []
        references = []
        self.model.eval()

        with torch.no_grad():
            # Iterate over the test set batches
            for src, tgt in tqdm(self.test_loader, desc="Testing Transformer", leave=True, unit="batch"):
                batch_size = tgt.size(0)

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
                        pred_ids = []
                        for token_id in pred_seq[1:]:
                            if token_id.item() == tgt_end_token:
                                break
                            # Make sure the index is within bounds.
                            pred_ids.append(token_id.item())
                            token = itos[token_id.item()] if token_id.item() < len(itos) else "<unk>"
                            pred_tokens.append(token)
                        # corpus bleu expects a list of reference sentences per candidate (list of lists)
                        candidates.append(pred_tokens)
                        candidates_ids.append(pred_ids)

                        # Process the reference translation (skipping the <sos> token).
                        ref_tokens = []
                        for token_id in tgt_seq[1:]:
                            if token_id.item() == tgt_end_token:
                                break
                            token = itos[token_id.item()] if token_id.item() < len(itos) else "<unk>"
                            ref_tokens.append(token)
                        # corpus_bleu expects a list of reference sentences per candidate (list of lists)
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

        with tqdm(total=4, desc="Calculating Metrics", leave=True, unit="metric") as pbar:
            bleu = corpus_bleu(references, candidates)
            pbar.update(1)

            precision, recall, f1 = self.calculate_bertscore(references, candidates)
            pbar.update(1)

            ref_oov_rate, pred_oov_rate = self.calculate_oov_rates(
                {"train_set_vocab": train_set, "test_set_vocab": test_set, "output_model_vocab": candidates_ids})
            pbar.update(1)

            unk_rate = self.calculate_unknown_rate(candidates)
            pbar.update(1)

        # Create a table
        table = Table(title="Evaluation Metrics")

        # Add columns
        table.add_column("Metric", justify="left", style="cyan", no_wrap=True)
        table.add_column("Value", justify="right", style="magenta")

        # Add rows with the results
        table.add_row("BLEU Score", f"{bleu:.4f}")
        table.add_row("Precision", f"{precision:.4f}")
        table.add_row("Recall", f"{recall:.4f}")
        table.add_row("F1 Score", f"{f1:.4f}")
        table.add_row("Reference OOV Rate (%)", f"{ref_oov_rate:.2f}")
        table.add_row("Prediction OOV Rate (%)", f"{pred_oov_rate:.2f}")
        table.add_row("Unknown Rate (%)", f"{unk_rate:.2f}")

        # Display the table
        console = Console()
        console.print(table)
