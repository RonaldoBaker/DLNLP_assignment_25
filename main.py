# Import dependencies
import torch
from functools import partial
from torch.utils.data import DataLoader
from torch import nn
from torch import optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from sklearn.model_selection import train_test_split
from src.preprocessor import Preprocessor
from src.custom_dataset import TranslationDataset, collate_fn
from src.model import Transformer
from src.model_trainer import TransformerTrainer

# Define the hyperparameters
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
random_seed = 7
embedding_size = 512
num_heads = 8
num_encoder_layers = 6
num_decoder_layers = 6
dropout = 0.1
max_len = 100
batch_size = 64
epochs = 100
lr = 0.0001

def main():
    # DATA PREPROCESSING
    # Read the text file
    lines = Preprocessor.read_text_file("data/spa-eng/spa.txt")
    print("Raw text data read")

    translation_dictionary = Preprocessor.create_parallel_data(text=lines, format="dict", save=False)
    print("Created dictionary of parallel sentences")

    # Tokenise the data
    tokenised_dataset = Preprocessor.create_tokenised_dataset(translation_dictionary)
    print("Dataset tokenised")

    # Create vocabulary
    eng_vocab, spa_vocab = Preprocessor.build_vocab(tokenised_dataset)
    print("Vocabulary built")

    # Convert the tokenised data to indices
    indexed_dataset = Preprocessor.numericalise(tokenised_dataset, eng_vocab, spa_vocab)
    print("Dataset indexed")

    # Create the custom dataset
    translation_dataset = TranslationDataset(indexed_dataset, eng_vocab, spa_vocab)
    print("Custom dataset created")

    # Split the data into train, validation and test sets
    train_dataset, val_dataset = train_test_split(translation_dataset, test_size=0.3, random_state=random_seed)
    val_dataset, test_dataset = train_test_split(val_dataset, test_size=1/3, random_state=random_seed)

    # Collate function is used to pad the sequences to the same length
    wrapped_collate_fn = partial(collate_fn, source_vocab=eng_vocab, target_vocab=spa_vocab)

    # Define the dataloader
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=wrapped_collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, collate_fn=wrapped_collate_fn)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, collate_fn=wrapped_collate_fn)
    print("Dataloader created")

    # MODEL TRAINING
    src_pad_index = eng_vocab["<pad>"]
    # Define the model
    transformer = Transformer(source_vocab_size=len(eng_vocab),
                              target_vocab_size=len(spa_vocab),
                              embedding_size=embedding_size,
                              nhead=num_heads,
                              num_encoder_layers=num_encoder_layers,
                              num_decoder_layers=num_decoder_layers,
                              dropout=dropout,
                              src_pad_index=src_pad_index,
                              max_len=max_len,
                              device=device).to(device)
    print("Model created")

    # Define the loss function, optimiser and scheduler
    loss_func = nn.CrossEntropyLoss(ignore_index=src_pad_index)
    optimiser = optim.Adam(transformer.parameters(), lr=lr)
    scheduler = ReduceLROnPlateau(optimiser, mode='min', factor=0.5, patience=3)
    print("Loss function, optimiser and scheduler defined")

    # Define model trainer
    trainer = TransformerTrainer(train_loader, val_loader, test_loader, epochs, optimiser, scheduler, loss_func, transformer, device)
    print("Model trainer created")

    # Train the model
    trainer.train()
    print("Model trained")

    # Ablation studies on different combinations of embeddings - subword, word and phrase embeddings
    # Train the model with different combinations of embeddings compared to baseline model with just word embeddings
    # Evaluate the model
    # Compare the results

    pass

if __name__ == '__main__':
    main()