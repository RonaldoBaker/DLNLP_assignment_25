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
"""
Ignore warnings for PyTorch nested tensors
This occurs due to the PyTorch version (2.3.0) being used, which is needed to use torchtext version 0.18.0
"""
import warnings
warnings.filterwarnings("ignore", message="The PyTorch API of nested tensors is in prototype stage") 

# Append project root to sys.path
project_root = os.path.join(os.path.dirname(__file__), "..")
if project_root not in sys.path:
    sys.path.append(project_root)

from src.preprocessor import Preprocessor
from src.custom_dataset import TokenDataset, MultiTokenDataset, collate_fn, collate_fn_multitokenisation
from src.models import Transformer, MultiSourceTransformer
from src.model_trainer import TransformerTrainer
from src.model_tester import TransformerTester
from utils.logger import Logger
from utils.config import config
import json

# Define the hyperparameters
random_seed = 7
embedding_size = 512
num_heads = 8
num_encoder_layers = 6
num_decoder_layers = 6
dropout = config.DROPOUT
max_len = 100
batch_size = 64
epochs = 50
lr = 0.0001

# Define parameter dictionary for logging
hyperparameters = {
    "fusion_type": config.FUSION_TYPE,
    "tokenisations": config.TOKENISATIONS,
    "model": config.MODEL,
    "random_seed": random_seed,
    "embedding_size": embedding_size,
    "num_heads": num_heads,
    "num_encoder_layers": num_encoder_layers,
    "num_decoder_layers": num_decoder_layers,
    "dropout": dropout,
    "max_len": max_len,
    "batch_size": batch_size,
    "epochs": epochs,
    "lr": lr,
}

# Set device
if torch.cuda.is_available():
    device_num = config.GPU
    if device_num == -1:
        device = torch.device("cpu")
    else:
        torch.cuda.set_device(device_num)
        device = torch.device(f"cuda:{device_num}")
else:
    device = torch.device("cpu")
print(f"Running on {device}")

# Python executable path
# This is the path to the Python executable in your conda environment
# /home/zceerba/.conda/envs/nlp/bin/python

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

    # Create parallel data from text file
    translation_dictionary = Preprocessor.create_parallel_data(text=lines, format="dict", save=False)
    print("Created dictionary of parallel sentences")

    # Tokenise the data
    tokenised_dictionaries = Preprocessor.create_tokenised_dataset(translation_dictionary)
    print("Dictionaries tokenised")

    # Create vocabulary
    vocabularies = Preprocessor.build_vocabularies(tokenised_dictionaries)
    src_vocab = vocabularies["src_word_vocab"]
    tgt_vocab = vocabularies["tgt_word_vocab"]
    print("Vocabularies built")

    # Convert the tokenised data to indices
    indexed_dictionaries = Preprocessor.numericalise(tokenised_dictionaries, vocabularies)
    print("Dictionaries indexed")

    # Save the preprocessed data
    if config.SAVE_DATA:
        with open(os.path.join(config.SAVE_PATH, "data", "indexed_dictionaries.json"), "w") as f:
            json.dump(indexed_dictionaries, f, indent=4)
        print("Indexed dictionaries saved to 'indexed_dictionaries.json'")

    # Create the custom dataset
    if config.MODEL == "single":
        token_dataset = TokenDataset(indexed_dictionaries, src_vocab, tgt_vocab, device)
    elif config.MODEL == "multi":
        chosen_tokenisations = ["src_" + tokenisation + "_ids" for tokenisation in config.TOKENISATIONS]
        token_dataset = MultiTokenDataset(indexed_dictionaries, chosen_tokenisations, device)
    print("Custom dataset created")

    # Split the data into train, validation and test sets
    train_set, val_set = train_test_split(token_dataset, test_size=0.3, random_state=random_seed)
    val_set, test_set = train_test_split(val_set, test_size=1/3, random_state=random_seed)

    # Define collate function based on the model type
    if config.MODEL == "single":
        wrapped_collate_fn = partial(collate_fn, source_vocab=src_vocab, target_vocab=tgt_vocab)
    elif config.MODEL == "multi":
        wrapped_collate_fn = partial(collate_fn_multitokenisation, source_vocab=src_vocab, target_vocab=tgt_vocab)

    # Create the data loaders
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, collate_fn=wrapped_collate_fn)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False, collate_fn=wrapped_collate_fn)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False, collate_fn=wrapped_collate_fn)
    print("Dataloader created")

    # MODEL TRAINING
    # Define pad index
    pad_index = src_vocab["<pad>"]

    # Create dictionary of vocab sizes
    vocab_sizes = {tokenisation: len(vocabularies[tokenisation.replace("_ids", "_vocab")]) 
                   for tokenisation in indexed_dictionaries[0].keys()
                    if tokenisation.endswith("_ids") and tokenisation.split("_")[1] in config.TOKENISATIONS}

    # Define the model
    if config.MODEL == "single":
        model = Transformer(source_vocab_size=len(src_vocab),
                            target_vocab_size=len(tgt_vocab),
                            embedding_size=embedding_size,
                            num_heads=num_heads,
                            num_encoder_layers=num_encoder_layers,
                            num_decoder_layers=num_decoder_layers,
                            dropout=dropout,
                            max_len=max_len,
                            pad_index=pad_index,
                            device=device).to(device)

    elif config.MODEL == "multi":
        model = MultiSourceTransformer(vocab_sizes=vocab_sizes,
                                       embedding_size=embedding_size,
                                       num_heads=num_heads,
                                       num_encoder_layers=num_encoder_layers,
                                       num_decoder_layers=num_decoder_layers,
                                       dropout=dropout,
                                       pad_index=pad_index,
                                       max_len=max_len,
                                       device=device,
                                       fusion_type=config.FUSION_TYPE).to(device)
    print("Model created")

    # Define the loss function, optimiser and scheduler
    loss_func = nn.CrossEntropyLoss(ignore_index=pad_index)
    optimiser = optim.Adam(model.parameters(), lr=lr)
    scheduler = ReduceLROnPlateau(optimiser, mode='min', factor=0.5, patience=1)
    print("Loss function, optimiser and scheduler defined")

    # Create logger - to create a new log directory for each run
    log_directory = Logger.set_log_dir(config.MODEL, hyperparameters)
    print(f"Log directory created: {log_directory}")

    # Define model trainer and tester
    trainer = TransformerTrainer(train_loader, val_loader, tgt_vocab, max_len, epochs, optimiser, scheduler, loss_func, model, device, log_directory)
    tester = TransformerTester(test_loader, model, device, log_directory)
    print("Model trainer and tester created")

    # Train the model
    print(f"TRAINING CONFIGURATION: MODEL = {config.MODEL} | FUSION TYPE = {config.FUSION_TYPE} | TOKENISATIONS = {config.TOKENISATIONS} | DROPOUT = {config.DROPOUT}")
    trainer.train(patience=3)
    print("Model trained")

    # Evaluate the model
    tester.evaluate(tgt_vocab=tgt_vocab, max_len=max_len, train_set=train_set, test_set=test_set, type="greedy")
    print("Model evaluated")

    # Log information for this run
    logger = Logger(trainer, tester, hyperparameters, config.MODEL)
    logger.log_dir = log_directory # set the log directory to the first one created
    logger.log_all()
    print("Model training and evaluation logged")
    print("DONE\n")

if __name__ == "__main__":
    main()