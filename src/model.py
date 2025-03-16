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
        self.device = device

    def make_src_mask(self, src):
        # src shape (batch size, src_seq_length)
        src_mask = src == self.src_pad_index
        return src_mask
    
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
        src_padding_mask = self.make_src_mask(src)
        trg_mask = self.transformer.generate_square_subsequent_mask(tgt_seq_length).to(self.device)
        output = self.transformer(src_embedded, tgt_embedded, src_key_padding_mask=src_padding_mask, tgt_mask=trg_mask)
        output = self.fc(output)
        return output