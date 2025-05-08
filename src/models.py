"""
This module contains the implementation of the Transformer model and its components.
"""
from typing import Union
import torch 
import torch.nn as nn

class BaseTransformer(nn.Module):
    """
    Base class for the Transformer model.
    """
    def __init__(self,
                 embedding_size: int,
                 num_heads: int,
                 num_encoder_layers: int,
                 num_decoder_layers: int,
                 dropout: float,
                 pad_index: int,
                 max_len: int,
                 device: str):
        super().__init__()
        self.embedding_size = embedding_size
        self.num_heads = num_heads
        self.num_encoder_layers = num_encoder_layers
        self.num_decoder_layers = num_decoder_layers
        self.dropout = dropout
        self.pad_index = pad_index
        self.max_len = max_len
        self.device = device
    

    def get_positional_embedding(self, sequences: dict[str, torch.tensor], type: str = "sequential") -> dict[str, torch.tensor]:
        """
        Get positional word embeddings for source and target sequences.
        Args:
            sequences (dict[str, torch.tensor]): Dictionary of sequences with tokenisation names as keys and tensors as values.
            type (str): Type of positional embedding to use. Options are "sequential" or "sinusoidal".
        Returns:
            dict[str, torch.tensor]: Dictionary of positional embeddings with tokenisation names as keys and tensors as values.
        """
        positional_embeddings = {}

        if type == "sequential":
        # Check that the number of input sequences is equal to the number of positional embedding layers
            assert len(sequences) == len(self.seq_pos_embedding_layers), \
                f"Number of input sequences ({len(sequences)}) does not match the number of positional embedding layers ({len(self.seq_pos_embedding_layers)})"
            
            for i, (tokenisation, sequence) in enumerate(sequences.items()):
                # Batch size and sequence length
                N, seq_length = sequence.shape

                # Using a simple sequential positional encoding
                positions = torch.arange(0, seq_length).unsqueeze(0).expand(N, seq_length).to(self.device)

                # Get positional embeddings using the tokenisation key
                positional_embeddings[tokenisation] = self.seq_pos_embedding_layers[i](positions)
        
        elif type == "sinusoidal":

            for i, (tokenisation, sequence) in enumerate(sequences.items()):
                # Batch size and sequence length
                N, seq_length = sequence.shape

                # Using a simple sequential positional encoding
                positions = torch.arange(0, seq_length, dtype=torch.float32, device=self.device).unsqueeze(1)
                div_term = torch.exp(torch.arange(0, self.embedding_size, 2, dtype=torch.float32, device=self.device)
                                    * -(torch.log(torch.tensor(10000.0, device=self.device)) / self.embedding_size))
                pe = torch.zeros(seq_length, self.embedding_size, device=self.device)
                pe[:, 0::2] = torch.sin(positions * div_term)
                pe[:, 1::2] = torch.cos(positions * div_term)
                pe = pe.unsqueeze(0).expand(N, seq_length, self.embedding_size)
                positional_embeddings[tokenisation] = pe

        else:
            raise ValueError("Invalid positional embedding type. Choose 'sequential' or 'sinusoidal'.")

        return positional_embeddings


    def make_src_key_padding_mask(self, src: Union[dict, torch.Tensor]) -> Union[dict, torch.Tensor]:
        """
        Create a key padding mask for the source sequences.
        Args:
            src (Union[dict, torch.Tensor]): Source sequences.
                If a dictionary, it should contain tokenisation names as keys and tensors as values.

        Returns:
            Union[dict, torch.Tensor]: Key padding mask for the source sequences.
            If a dictionary, it will contain tokenisation names as keys and masks as values.
        """
        if isinstance(src, dict):
            masks = {}
            for tokenisation, tensor in src.items():
                mask = tensor == self.pad_index
                masks[tokenisation] = mask
            return masks
        else:
            # src shape (batch size, src_seq_length)
            mask = src == self.pad_index
            return mask


    def make_tgt_key_padding_mask(self, tgt: torch.Tensor) -> torch.Tensor:
        """
        Create a key padding mask for the target sequences.
        Args:
            tgt (torch.Tensor): Target sequences.
        Returns:
            torch.Tensor: Key padding mask for the target sequences.
        """
        # tgt shape (batch size, tgt_seq_length)
        mask = tgt == self.pad_index
        return mask


class Transformer(BaseTransformer):
    """
    Baseline single-source Transformer model.
    """
    def __init__(self,
                 source_vocab_size: int,
                 target_vocab_size: int,
                 embedding_size: int,
                 num_heads: int,
                 num_encoder_layers: int,
                 num_decoder_layers: int,
                 dropout: float,
                 pad_index: int,
                 max_len: int,
                 device: str):
        super().__init__(embedding_size, num_heads, num_encoder_layers, num_decoder_layers, dropout, pad_index, max_len, device)
        self.source_vocab_size = source_vocab_size
        self.target_vocab_size = target_vocab_size

        # Word enbeddings for source and target
        self.src_word_embedding = nn.Embedding(self.source_vocab_size, self.embedding_size)
        self.tgt_word_embedding = nn.Embedding(self.target_vocab_size, self.embedding_size)

        # Positional embeddings for source and target
        self.seq_pos_embedding_layers = nn.ModuleList([
            nn.Embedding(self.max_len, self.embedding_size) for _ in range(2)])

        # Transformer model
        self.transformer = nn.Transformer(d_model=self.embedding_size,
                                          nhead=self.num_heads,
                                          num_encoder_layers=self.num_encoder_layers,
                                          num_decoder_layers=self.num_decoder_layers,
                                          dropout=self.dropout,
                                          activation='relu',
                                          batch_first=True)

        # Fully connected layer
        self.fc = nn.Linear(self.embedding_size, target_vocab_size)
        self.dropout = nn.Dropout(self.dropout)
        self.device = self.device


    def forward(self, src: torch.Tensor, tgt: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for the Transformer model.
        Args:
            src (torch.Tensor): Source sequences.
            tgt (torch.Tensor): Target sequences.

        Returns:
            torch.Tensor: Output of the Transformer model.
        """
        # Get positional encodings
        positional_embeddings = self.get_positional_embedding({"src_word_ids": src, "tgt_word_ids": tgt})

        src_embedded = self.dropout(self.src_word_embedding(src) + positional_embeddings["src_word_ids"])
        tgt_embedded = self.dropout(self.tgt_word_embedding(tgt) + positional_embeddings["tgt_word_ids"])

        # So that the transformer knows where the padding is
        src_padding_mask = self.make_src_key_padding_mask(src)
        tgt_padding_mask = self.make_tgt_key_padding_mask(tgt)
        tgt_mask = self.transformer.generate_square_subsequent_mask(tgt.shape[1], device=self.device, dtype=torch.bool)

        # Transformer forward pass
        output = self.transformer(src=src_embedded,
                                  tgt=tgt_embedded,
                                  tgt_mask=tgt_mask,
                                  src_key_padding_mask=src_padding_mask,
                                  tgt_key_padding_mask=tgt_padding_mask)
        # Fully connected layer
        output = self.fc(output)

        return output


class AttentionFusion(nn.Module):
    """
    Attention-based fusion layer for multiple tokenisation methods.
    """
    def __init__(self, embedding_size: int, num_heads: int, layer_names: list[str]):
        super(AttentionFusion, self).__init__()
        # Initialize the attention layers
        self.single_attention_layer = nn.MultiheadAttention(embed_dim=embedding_size, num_heads=num_heads, batch_first=True)

        self.multi_attention_layers = nn.ModuleDict({
            name: nn.MultiheadAttention(embed_dim=embedding_size, num_heads=num_heads, batch_first=True)
             for name in layer_names})
        self.layer_names = layer_names

    def forward(self, srcs: dict[str, torch.tensor], type: str = "single", lpes: dict[str, torch.tensor] = None) -> torch.tensor:
        """
        Forward pass for the attention fusion layer.
        Args:
            srcs (dict[str, torch.tensor]): Dictionary of source sequences with tokenisation names as keys and tensors as values.
            type (str): Type of attention fusion to use. Options are "single", "multi" or "lattice".
            lpes (dict[str, torch.tensor]): Dictionary of lattice positional encodings with tokenisation names as keys and tensors as values.
        Returns:
            torch.Tensor: Fused output of the attention fusion layer.
        """
        attention_outputs = [] # Empty list to store attention outputs
        if type == "single" or type == "lattice":
            # Calculate attention between the base tokenisation and the other tokenisations
            for name in self.layer_names:
                query = srcs["src_word_ids"]
                key = srcs[name]
                value = srcs[name]

                # Lattice Positional Encodings
                if type == "lattice":
                    assert lpes is not None, "Lattice positional encodings are required for lattice fusion"
                    # Add the lattice positional encodings to the query, key and value
                    query = srcs["src_word_ids"] + lpes["src_word_lpes"]
                    key = srcs[name] + lpes[name.replace("ids", "lpes")]
                    value = srcs[name] + lpes[name.replace("ids", "lpes")]

                attention_output, _ = self.single_attention_layer(query=query, key=key, value=value)
                attention_outputs.append(attention_output)

        elif type == "multi":
            for tokenisation in self.multi_attention_layers.keys():
                attention_output, _ = self.multi_attention_layers[tokenisation](query=srcs["src_word_ids"],
                                                                                key=srcs[tokenisation], value=srcs[tokenisation])
                attention_outputs.append(attention_output)

        else:
            raise ValueError("Invalid attention type. Choose 'single' or 'multi'.")
        
        # Sum the attention outputs
        fused_output = sum(attention_outputs) # (B, S_l, D)
        return fused_output


class MultiSourceTransformer(BaseTransformer):
    """
    Multi-source Transformer model for multiple tokenisation methods.
    This model uses attention-based fusion to combine the outputs of multiple encoders.
    """
    def __init__(self, vocab_sizes: dict[str, int],
                 embedding_size: int,
                 num_heads: int,
                 num_encoder_layers: int,
                 num_decoder_layers: int,
                 dropout: float,
                 pad_index: int,
                 max_len: int,
                 device: str,
                 fusion_type: str,
                 pe_type: str = "sequential",
                 additional_positional_embedding: bool = True):
        super().__init__(embedding_size, num_heads, num_encoder_layers, num_decoder_layers, dropout, pad_index, max_len, device)
        self.vocab_sizes = vocab_sizes
        self.num_sources = len(vocab_sizes)
        self.fusion_type = fusion_type
        self.pe_type = pe_type
        self.additional_positional_embedding = additional_positional_embedding

        # Create embeddings and encoders dynamically
        self.seq_pos_embedding_layers = nn.ModuleList([
            nn.Embedding(max_len, embedding_size) for _ in range(self.num_sources)]) # Positional embeddings for each tokenisation
    
        # Embedding layers for each tokenisation
        self.embedding_layers = nn.ModuleDict(
            {tokenisation: nn.Embedding(vocab_size, self.embedding_size) for tokenisation, vocab_size in self.vocab_sizes.items()}) 
        

        self.encoders = nn.ModuleList([
            nn.TransformerEncoder(
                nn.TransformerEncoderLayer(d_model=self.embedding_size, nhead=self.num_heads, dropout=self.dropout, batch_first=True),
                num_layers=self.num_encoder_layers) for _ in range(self.num_sources)
        ]) # Transformer encoders for each tokenisation

        # Using word-level tokenisation as the output
        self.decoder = nn.TransformerDecoder(
            nn.TransformerDecoderLayer(d_model=self.embedding_size, nhead=self.num_heads, dropout=self.dropout, batch_first=True),
            num_layers=self.num_decoder_layers)

        # Output layer
        self.fc_out = nn.Linear(self.embedding_size, self.vocab_sizes["tgt_word_ids"])

        # Dropout layer
        self.dropout = nn.Dropout(self.dropout)

        # Create the attention fusion layer
        layer_names = [name for name in self.vocab_sizes.keys() if name != "src_word_ids" and name != "tgt_word_ids"]
        self.fuser = AttentionFusion(embedding_size, self.num_heads, layer_names)


    def generate_square_subsequent_mask(self, size: int) -> torch.Tensor:
        """
        Generate a square subsequent mask for the decoder.
        Args:
            size (int): The size of the mask.
        Returns:
            (torch.Tensor): A square subsequent mask.
        """
        mask = torch.triu(torch.full((size, size), float("-inf")), diagonal=1).bool()
        return mask


    def forward(self, srcs: dict[str, torch.Tensor], tgt: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for the multi-source transformer model.
        Args:
            srcs (dict[str, torch.Tensor]): Dictionary of source sequences with tokenisation names as keys and tensors as values.
            tgt (torch.Tensor): Target sequences.
        Returns:
            torch.Tensor: Output of the multi-source transformer model.
        """
        # Separate the lattice positional encodings from the source tokenisations
        src_lpes = None
        if self.fusion_type == "lattice":
            src_lpes = {key: srcs[key] for key in srcs.keys() if key.endswith("_lpes")}
        src_sequences = {key: srcs[key] for key in srcs.keys() if not key.endswith("_lpes")}
        src_sequences.update({"tgt_word_ids": tgt})

        if self.additional_positional_embedding:
            positional_embeddings = self.get_positional_embedding(src_sequences, self.pe_type)
        else:
            # Just adding a positional encoding to the word tokens
            src_word_sequences = {key: srcs[key] for key in srcs.keys() if key == "src_word_ids"}
            positional_embeddings = self.get_positional_embedding(src_word_sequences, self.pe_type)
        
        # Get the embeddings for source tokenisations and target tokenisation
        embeddings = {tokenisation: self.embedding_layers[tokenisation](tensor) for tokenisation, tensor in src_sequences.items()}

        # Add positional embeddings to the source and target embedding and apply dropout
        # If there is no corresponding positional embedding, just add 0
        embeddings = {
            tokenisation: self.dropout(embedding + positional_embeddings.get(tokenisation, 0)) for tokenisation, embedding in embeddings.items()
        }

        # Make source key padding masks
        src_key_padding_masks = self.make_src_key_padding_mask(srcs)

        # Split the embeddings dictionary into source and target embeddings
        tgt_embedding = embeddings.pop("tgt_word_ids") # Remove target tokenisation from the source embeddings
        src_embeddings = embeddings 

        # Encoder forward pass
        encoded_outputs = {}
        for i, (tokenisation, tensor) in enumerate(src_embeddings.items()):
            enc_output = self.encoders[i](src=tensor, src_key_padding_mask=src_key_padding_masks[tokenisation])
            encoded_outputs[tokenisation] = enc_output

        # Attention-based fusion if there is more than one tokenisation method,
        # if not return the regular encoder output for word tokenisation
        fused_output = encoded_outputs["src_word_ids"] if len(encoded_outputs) == 1 else self.fuser(encoded_outputs, self.fusion_type, src_lpes)

        # Decoder forward pass
        tgt_key_padding_mask = self.make_tgt_key_padding_mask(tgt)
        tgt_mask = self.generate_square_subsequent_mask(tgt.shape[1]).to(self.device)
        dec_output = self.decoder(tgt=tgt_embedding,
                                  memory=fused_output,
                                  tgt_mask=tgt_mask,
                                  tgt_key_padding_mask=tgt_key_padding_mask)

        # Output layer
        output = self.fc_out(dec_output)

        return output
