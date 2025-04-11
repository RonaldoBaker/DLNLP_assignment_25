import torch 
import torch.nn as nn

class Transformer(nn.Module):
    def __init__(self,
                 source_vocab_size,
                 target_vocab_size,
                 embedding_size,
                 nhead,
                 num_encoder_layers,
                 num_decoder_layers,
                 dropout,
                 src_pad_index,
                 tgt_pad_index,
                 max_len,
                 device):
        super(Transformer, self).__init__()
        # Word enbeddings for source and target
        self.src_word_embedding = nn.Embedding(source_vocab_size, embedding_size)
        self.tgt_word_embedding = nn.Embedding(target_vocab_size, embedding_size)

        # Positional embeddings for source and target
        self.src_pos_embedding = nn.Embedding(max_len, embedding_size)
        self.tgt_pos_embedding = nn.Embedding(max_len, embedding_size)

        # Transformer model
        self.transformer = nn.Transformer(d_model=embedding_size,
                                          nhead=nhead,
                                          num_encoder_layers=num_encoder_layers,
                                          num_decoder_layers=num_decoder_layers,
                                          dropout=dropout,
                                          activation='relu',
                                          batch_first=True)
        
        # Fully connected layer
        self.fc = nn.Linear(embedding_size, target_vocab_size)
        self.dropout = nn.Dropout(dropout)

        # Other attributes for forward pass
        self.src_pad_index = src_pad_index
        self.tgt_pad_index = tgt_pad_index
        self.device = device

    def make_src_key_padding_mask(self, src):
        # src shape (batch size, src_seq_length)
        mask = src == self.src_pad_index
        return mask
    
    def make_tgt_key_padding_mask(self, tgt):
        # src shape (batch size, src_seq_length)
        mask = tgt == self.tgt_pad_index
        return mask
    
    def forward(self, src, tgt):
        N, src_seq_length = src.shape
        N, tgt_seq_length = tgt.shape

        # Positional encodings
        src_positions = torch.arange(0, src_seq_length).unsqueeze(0).expand(
            N, src_seq_length).to(self.device)
        
        tgt_positions = torch.arange(0, tgt_seq_length).unsqueeze(0).expand(
            N, tgt_seq_length).to(self.device)

        src_embedded = self.dropout(self.src_word_embedding(src) + self.src_pos_embedding(src_positions))
        tgt_embedded = self.dropout(self.tgt_word_embedding(tgt) + self.tgt_pos_embedding(tgt_positions))

        # So that the transformer knows where the padding is
        src_padding_mask = self.make_src_key_padding_mask(src).to(torch.float32)
        tgt_padding_mask = self.make_tgt_key_padding_mask(tgt).to(torch.float32)
        tgt_mask = self.transformer.generate_square_subsequent_mask(tgt_seq_length).to(self.device)

        # Transformer forward pass
        output = self.transformer(src=src_embedded,
                                  tgt=tgt_embedded,
                                  tgt_mask=tgt_mask,
                                  src_key_padding_mask=src_padding_mask,
                                  tgt_key_padding_mask=tgt_padding_mask)

        # Fully connected layer
        output = self.fc(output)

        return output


class MultiSourceTransformer(nn.Module):
    def __init__(self, vocab_sizes: dict[str, int], embedding_size, nhead, num_encoder_layers, num_decoder_layers, dropout, max_len, device, pad_index):
        super(MultiSourceTransformer, self).__init__()
        self.num_sources = len(vocab_sizes)
        self.device = device
        self.pad_index = pad_index

        # Positional embeddings for source and target
        self.src_pos_embedding = nn.Embedding(max_len, embedding_size)
        self.tgt_pos_embedding = nn.Embedding(max_len, embedding_size)

        # Create embeddings and encoders dynamically
        self.src_embeddings = nn.ModuleDict(
            {tokenisation: nn.Embedding(vocab_size, embedding_size) for tokenisation, vocab_size in vocab_sizes.items()})
        self.encoders = nn.ModuleList([
            nn.TransformerEncoder(
                nn.TransformerEncoderLayer(d_model=embedding_size, nhead=nhead, dropout=dropout, batch_first=True), num_layers=num_encoder_layers
            ) for _ in range(self.num_sources)
        ])

        # Using word-level tokenisation as the output
        self.tgt_embedding = nn.Embedding(vocab_sizes["tgt_word_ids"], 512)
        self.decoder = nn.TransformerDecoder(
            nn.TransformerDecoderLayer(d_model=embedding_size, nhead=nhead, dropout=dropout, batch_first=True), num_layers=num_decoder_layers
        )

        # Output layer
        self.fc_out = nn.Linear(512, vocab_sizes["tgt_word_ids"])


    def cross_attention_fusion(self, tensors: list[torch.tensor], fused_seq_len: int):
        """
        Args:
            tensors: list of Tensors with shape (B, S_i, D)
            fused_seq_len: number of query tokens to use
            feature_dim: if not provided, inferred from input
        Returns:
            fused tensor: shape (B, N * fused_seq_len, D)
        """
        B = tensors[0].shape[0]
        D = tensors[0].shape[2]
        N = len(tensors)

        # Create learnable queries (1, fused_seq_len, D)
        query_tokens = torch.randn(1, fused_seq_len, D, device=tensors[0].device)
        query_tokens = query_tokens.expand(B, -1, -1)  # (B, fused_seq_len, D)

        fused = []

        for x in tensors:  # x: (B, S, D)
            # Attention scores: (B, fused_seq_len, S)
            attn_scores = torch.matmul(query_tokens, x.transpose(1, 2)) / (D ** 0.5)
            attn_weights = torch.softmax(attn_scores, dim=-1)
            # Weighted sum: (B, fused_seq_len, D)
            out = torch.matmul(attn_weights, x)
            fused.append(out)

        # Concatenate across tensors: (B, N * fused_seq_len, D)
        return torch.cat(fused, dim=1)


    def make_src_key_padding_mask(self, srcs: dict[str, torch.Tensor]):
        # src shape (batch size, src_seq_length)
        masks = {}
        for tokenisation, tensor in srcs.items():
            mask = tensor == self.pad_index
            masks[tokenisation] = mask
        return masks


    def make_tgt_key_padding_mask(self, tgt):
        # tgt shape (batch size, tgt_seq_length)
        mask = tgt == self.pad_index
        return mask


    def generate_square_subsequent_mask(self, size: int) -> torch.Tensor:
        """
        Generate a square subsequent mask for the decoder.
        Args:
            size: The size of the mask.
        Returns:
            A square subsequent mask.
        """
        mask = torch.triu(torch.full((size, size), float("-inf")), diagonal=1).float()
        return mask


    def forward(self, srcs: dict[str, torch.Tensor], tgt: torch.Tensor):
        # Get positional encodings using the word-level tokenisation
        src_word_tokenisation = srcs["src_word_ids"]
        tgt_word_tokenisation = tgt

        N, src_seq_length = src_word_tokenisation.shape
        N, tgt_seq_length = tgt_word_tokenisation.shape

        src_positions = torch.arange(0, src_seq_length).unsqueeze(0).expand(
            N, src_seq_length).to(self.device)

        tgt_positions = torch.arange(0, tgt_seq_length).unsqueeze(0).expand(
            N, tgt_seq_length).to(self.device)

        # Get the embeddings for source tokenisations and target tokenisation
        src_embeddings = {tokenisation: self.src_embeddings[tokenisation](tensor) for tokenisation, tensor in srcs.items()}
        tgt_embedding = self.tgt_embedding(tgt_word_tokenisation)
        src_positional_embedding = self.src_pos_embedding(src_positions) # Positional embeddings
        tgt_positional_embedding= self.tgt_pos_embedding(tgt_positions)

        # Add positional embeddings to the source and target word embeddings
        src_embeddings["src_word_ids"] += src_positional_embedding
        tgt_embedding += tgt_positional_embedding

        # Make source key padding masks
        src_key_padding_masks = self.make_src_key_padding_mask(srcs)

        # Encoder forward pass
        encoded_outputs = []
        for i, (tokenisation, tensor) in enumerate(src_embeddings.items()):
            enc_output = self.encoders[i](src=tensor, src_key_padding_mask=src_key_padding_masks[tokenisation])
            encoded_outputs.append(enc_output)

        # Concatenate the outputs from all sources
        fused_outputs = self.cross_attention_fusion(encoded_outputs, tgt_seq_length)

        # Decoder forward pass
        tgt_key_padding_mask = self.make_tgt_key_padding_mask(tgt).to(torch.float32)
        tgt_mask = self.generate_square_subsequent_mask(tgt_seq_length).to(self.device)
        dec_output = self.decoder(tgt=tgt_embedding,
                                  memory=fused_outputs,
                                  tgt_mask=tgt_mask,
                                  tgt_key_padding_mask=tgt_key_padding_mask)

        # Output layer
        output = self.fc_out(dec_output)

        return output
