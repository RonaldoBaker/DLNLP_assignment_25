# Import dependencies
import os
import sys
import torch
from functools import partial
from torch.utils.data import DataLoader
from torch import nn
from torch import optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from sklearn.model_selection import train_test_split

# Append project root to sys.path
project_root = os.path.join(os.path.dirname(__file__), "..")
if project_root not in sys.path:
    sys.path.append(project_root)

from src.preprocessor import Preprocessor
from src.custom_dataset import TokenDataset, collate_fn
from src.models import Transformer
from src.model_trainer import TransformerTrainer
from src.model_tester import TransformerTester
from utils.config import config

# Define the hyperparameters
random_seed = 7
embedding_size = 512
num_heads = 8
num_encoder_layers = 6
num_decoder_layers = 6
dropout = 0.2
max_len = 100
batch_size = 64
epochs = 50
lr = 0.0001

# /home/zceerba/.conda/envs/nlp/bin/python

# Set device
if torch.cuda.is_available():
    device_num = config.GPU
    torch.cuda.set_device(device_num)
    device = torch.device(f"cuda:{device_num}")
else:
    device = torch.device("cpu")
print(f"Running on {device}")

def main():
    # Set random seed for reproducibility
    torch.manual_seed(random_seed)
    torch.cuda.manual_seed_all(random_seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # DATA PREPROCESSING
    # Read the text file
    lines = Preprocessor.read_text_file("data/spa-eng/spa.txt")
    print("Raw text data read")

    translation_dictionary = Preprocessor.create_parallel_data(text=lines, format="dict", save=False)
    print("Created dictionary of parallel sentences")

    # Tokenise the data
    tokenised_dictionaries = Preprocessor.create_tokenised_dataset(translation_dictionary)
    print("Dataset tokenised")

    # Create vocabulary
    vocabularies = Preprocessor.build_vocabularies(tokenised_dictionaries)
    eng_vocab = vocabularies["src_word_vocab"]
    spa_vocab = vocabularies["tgt_word_vocab"]
    print("Vocabulary built")

    # Convert the tokenised data to indices
    indexed_dictionaries = Preprocessor.numericalise(tokenised_dictionaries, vocabularies)
    print("Dataset indexed")

    # Create the custom dataset
    token_dataset = TokenDataset(indexed_dictionaries, eng_vocab, spa_vocab, device)
    print("Custom dataset created")

    # Split the data into train, validation and test sets
    train_set, val_set = train_test_split(token_dataset, test_size=0.3, random_state=random_seed)
    val_set, test_set = train_test_split(val_set, test_size=1/3, random_state=random_seed)

    # Collate function is used to pad the sequences to the same length
    wrapped_collate_fn = partial(collate_fn, source_vocab=eng_vocab, target_vocab=spa_vocab)

    # Define the dataloader
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, collate_fn=wrapped_collate_fn)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False, collate_fn=wrapped_collate_fn)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False, collate_fn=wrapped_collate_fn)
    print("Dataloader created")

    # MODEL TRAINING
    src_pad_index = eng_vocab["<pad>"]
    # Define the model
    transformer = Transformer(source_vocab_size=len(eng_vocab),
                              target_vocab_size=len(spa_vocab),
                              embedding_size=embedding_size,
                              num_heads=num_heads,
                              num_encoder_layers=num_encoder_layers,
                              num_decoder_layers=num_decoder_layers,
                              dropout=dropout,
                              pad_index=src_pad_index,
                              max_len=max_len,
                              device=device).to(device)
    print("Model created")

    # Define the loss function, optimiser and scheduler
    loss_func = nn.CrossEntropyLoss(ignore_index=src_pad_index)
    optimiser = optim.Adam(transformer.parameters(), lr=lr)
    scheduler = ReduceLROnPlateau(optimiser, mode='min', factor=0.5, patience=1)
    print("Loss function, optimiser and scheduler defined")

    # Define model trainer
    trainer = TransformerTrainer(train_loader, val_loader, test_loader, epochs, optimiser, scheduler, loss_func, transformer, device)
    tester = TransformerTester(test_loader, transformer, device)
    print("Model trainer and tester created")

    # Train the model
    trainer.train(patience=2)
    print("Model trained")

    # Plot loss curves
    trainer.plot_loss_curves(epoch_resolution=1, path=config.SAVE_FILEPATH + "figures/loss_curves.png")

    # Evaluate the model
    tester.evaluate(tgt_vocab=spa_vocab, max_len=max_len, train_set=train_set, test_set=test_set, type="greedy")
    print("Model evaluated")

if __name__ == '__main__':
    main()