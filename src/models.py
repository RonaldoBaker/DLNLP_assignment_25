from typing import Union
import torch 
import torch.nn as nn

class BaseTransformer(nn.Module):
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


    def get_positional_word_embeddings(self, src: Union[dict, torch.Tensor], tgt: torch.Tensor):
        """
        Get positional word embeddings for source and target sequences.
        Args:
            src: Source sequence (dict of different source tokenisations or tensor).
            tgt: Target sequence (tensor).
        Returns:
            src_embeddings: Source embeddings with positional information from word tokenisations.
            tgt_embeddings: Target embeddings with positional information from word tokenisations.
        """
        if isinstance(src, dict):
            src = src["src_word_ids"]

        N, src_seq_length = src.shape # batch size, source sequence length
        N, tgt_seq_length = tgt.shape # batch size, target sequence length

        # Get positional encodings using the word-level tokenisation
        src_positions = torch.arange(0, src_seq_length).unsqueeze(0).expand(
            N, src_seq_length).to(self.device)

        tgt_positions = torch.arange(0, tgt_seq_length).unsqueeze(0).expand(
            N, tgt_seq_length).to(self.device)

        # Positional embeddings
        src_positional_embedding = self.src_pos_embedding(src_positions)
        tgt_positional_embedding = self.tgt_pos_embedding(tgt_positions)

        return src_positional_embedding, tgt_positional_embedding


    def make_src_key_padding_mask(self, src: Union[dict, torch.Tensor]) -> Union[dict, torch.Tensor]:
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


    def make_tgt_key_padding_mask(self, tgt):
        # tgt shape (batch size, tgt_seq_length)
        mask = tgt == self.pad_index
        return mask


class Transformer(BaseTransformer):
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
        self.src_pos_embedding = nn.Embedding(self.max_len, self.embedding_size)
        self.tgt_pos_embedding = nn.Embedding(self.max_len, self.embedding_size)

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


    def forward(self, src, tgt):
        # Get positional encodings
        src_positional_embedding, tgt_positional_embedding = self.get_positional_word_embeddings(src, tgt)

        src_embedded = self.dropout(self.src_word_embedding(src) + src_positional_embedding)
        tgt_embedded = self.dropout(self.tgt_word_embedding(tgt) + tgt_positional_embedding)

        # So that the transformer knows where the padding is
        src_padding_mask = self.make_src_key_padding_mask(src)
        tgt_padding_mask = self.make_tgt_key_padding_mask(tgt)
        # assert src_padding_mask.shape == (src_embedded.shape[0], src_embedded.shape[1]), "src_key_padding_mask shape mismatch"
        # assert tgt_padding_mask.shape == (tgt_embedded.shape[0], tgt_embedded.shape[1]), "tgt_key_padding_mask shape mismatch"
        tgt_mask = self.transformer.generate_square_subsequent_mask(tgt.shape[1], device=self.device, dtype=torch.float32)
        # assert tgt_mask.shape == (tgt.shape[1], tgt.shape[1]), "tgt_mask shape mismatch"

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
    def __init__(self, embedding_size: int, num_heads: int, layer_names: list[str]):
        super(AttentionFusion, self).__init__()
        self.single_attention_layer = nn.MultiheadAttention(embed_dim=embedding_size, num_heads=num_heads, batch_first=True)

        self.multi_attention_layers = nn.ModuleDict({
            name: nn.MultiheadAttention(embed_dim=embedding_size, num_heads=num_heads, batch_first=True)
             for name in layer_names})

    def forward(self, srcs: dict[str, torch.tensor], type: str = "single"):
        attention_outputs = [] # Empty list to store attention outputs
        if type == "single":
            # Calculate attention between the base tokenisation and the other tokenisations
            for _, encoded_output in srcs.items():
                attention_output, _ = self.single_attention_layer(query=srcs["src_word_ids"], key=encoded_output, value=encoded_output)
                attention_outputs.append(attention_output)

        elif type == "multi":
            for tokenisation in self.multi_attention_layers.keys():
                attention_output, _ = self.multi_attention_layers[tokenisation](query=srcs["src_word_ids"], key=srcs[tokenisation], value=srcs[tokenisation])
                attention_outputs.append(attention_output)

        else:
            raise ValueError("Invalid attention type. Choose 'single' or 'multi'.")
        
        # Sum the attention outputs
        fused_output = sum(attention_outputs) # (B, S_l, D)
        return fused_output


class MultiSourceTransformer(BaseTransformer):
    def __init__(self, vocab_sizes: dict[str, int],
                 embedding_size: int,
                 num_heads: int,
                 num_encoder_layers: int,
                 num_decoder_layers: int,
                 dropout: float,
                 pad_index: int,
                 max_len: int,
                 device: str,
                 fusion_type: str):
        super().__init__(embedding_size, num_heads, num_encoder_layers, num_decoder_layers, dropout, pad_index, max_len, device)
        self.vocab_sizes = vocab_sizes
        self.num_sources = len(vocab_sizes)
        self.fusion_type = fusion_type

        # Positional embeddings for source and target
        self.src_pos_embedding = nn.Embedding(max_len, embedding_size)
        self.tgt_pos_embedding = nn.Embedding(max_len, embedding_size)

        # Create embeddings and encoders dynamically
        self.src_embeddings = nn.ModuleDict(
            {tokenisation: nn.Embedding(vocab_size, self.embedding_size) for tokenisation, vocab_size in self.vocab_sizes.items()})
        self.encoders = nn.ModuleList([
            nn.TransformerEncoder(
                nn.TransformerEncoderLayer(d_model=self.embedding_size, nhead=self.num_heads, dropout=self.dropout, batch_first=True), num_layers=self.num_encoder_layers
            ) for _ in range(self.num_sources)
        ])

        # Using word-level tokenisation as the output
        self.tgt_embedding = nn.Embedding(self.vocab_sizes["tgt_word_ids"], self.embedding_size)
        self.decoder = nn.TransformerDecoder(
            nn.TransformerDecoderLayer(d_model=self.embedding_size, nhead=self.num_heads, dropout=self.dropout, batch_first=True), num_layers=self.num_decoder_layers
        )

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
            size: The size of the mask.
        Returns:
            A square subsequent mask.
        """
        mask = torch.triu(torch.full((size, size), float("-inf")), diagonal=1).bool()
        return mask


    def forward(self, srcs: dict[str, torch.Tensor], tgt: torch.Tensor):
        # # Get positional encodings using the word-level tokenisation
        src_positional_embedding, tgt_positional_embedding = self.get_positional_word_embeddings(srcs, tgt)

        # Get the embeddings for source tokenisations and target tokenisation
        src_embeddings = {tokenisation: self.src_embeddings[tokenisation](tensor) for tokenisation, tensor in srcs.items()}
        tgt_embedding = self.tgt_embedding(tgt)

        # Add positional embeddings to the source and target word embeddings
        src_embeddings["src_word_ids"] = self.dropout(src_embeddings["src_word_ids"] + src_positional_embedding)
        tgt_embedding = self.dropout(tgt_embedding + tgt_positional_embedding)

        # Make source key padding masks
        src_key_padding_masks = self.make_src_key_padding_mask(srcs)

        # Encoder forward pass
        encoded_outputs = {}
        for i, (tokenisation, tensor) in enumerate(src_embeddings.items()):
            enc_output = self.encoders[i](src=tensor, src_key_padding_mask=src_key_padding_masks[tokenisation])
            encoded_outputs[tokenisation] = enc_output

        # Attention-based fusion if there is more than one tokenisation method,
        # if not return the regular encoder output for word tokenisation
        fused_output = encoded_outputs["src_word_ids"] if len(encoded_outputs) == 1 else self.fuser(encoded_outputs, self.fusion_type)

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
