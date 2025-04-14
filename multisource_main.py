# Import dependencies
import torch
from functools import partial
from torch.utils.data import DataLoader
from torch import nn
from torch import optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from sklearn.model_selection import train_test_split
from src.preprocessor import Preprocessor
from src.custom_dataset import MultiTokenDataset, collate_fn_multitokenisation
from src.models import MultiSourceTransformer
from src.model_trainer import TransformerTrainer

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
    device_num = 5
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
    tokenised_dataset = Preprocessor.create_tokenised_dataset(translation_dictionary)
    print("Dataset tokenised")

    # Create vocabulary
    vocabularies = Preprocessor.build_vocabularies(tokenised_dataset)
    print("Vocabulary built")

    # Convert the tokenised data to indices
    indexed_dataset = Preprocessor.numericalise(tokenised_dataset, vocabularies)
    print("Dataset indexed")

    # Create the custom dataset
    chosen_tokenisations = ["src_word_ids"]
    token_dataset = MultiTokenDataset(indexed_dataset, chosen_tokenisations, device)
    print("Custom dataset created")

    # Split the data into train, validation and test sets
    train_dataset, val_dataset = train_test_split(token_dataset, test_size=0.3, random_state=random_seed)
    val_dataset, test_dataset = train_test_split(val_dataset, test_size=1/3, random_state=random_seed)

    wrapped_collate_fn = partial(collate_fn_multitokenisation, source_vocab=vocabularies["src_word_vocab"], target_vocab=vocabularies["tgt_word_vocab"])

    # Define the dataloader
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=wrapped_collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, collate_fn=wrapped_collate_fn)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, collate_fn=wrapped_collate_fn)
    print("Dataloader created")

    # MODEL TRAINING
    vocab_sizes = {
    "src_word_ids": len(vocabularies["src_word_vocab"]),
    "tgt_word_ids": len(vocabularies["tgt_word_vocab"])
    }

    pad_index = vocabularies["src_word_vocab"]["<pad>"] # Pad index is the same for all vocabs

    model = MultiSourceTransformer(vocab_sizes=vocab_sizes,
                                embedding_size=embedding_size,
                                num_heads=num_heads,
                                num_encoder_layers=num_encoder_layers,
                                num_decoder_layers=num_decoder_layers,
                                dropout=dropout,
                                max_len=max_len,
                                device=device,
                                pad_index=pad_index,
                                fusion_type="single").to(device)
    print("Model created")

    loss_func = nn.CrossEntropyLoss(ignore_index=pad_index)
    optimiser = optim.Adam(model.parameters(), lr=lr)
    scheduler = ReduceLROnPlateau(optimiser, mode='min', factor=0.5, patience=1)
    print("Loss function, optimiser and scheduler defined")

    # Define model trainer
    trainer = TransformerTrainer(train_loader, val_loader, test_loader, epochs, optimiser, scheduler, loss_func, model, device)
    print("Model trainer created")

    # Train the model
    trainer.train(patience=2)
    print("Model trained")

    # Evaluate the model
    trainer.evaluate_bleu(tgt_vocab=vocabularies["tgt_word_vocab"], max_len=max_len, type="greedy")
    print("Model evaluated")

    # Plot loss curves
    trainer.plot_loss_curves(epoch_resolution=1, path="/home/zceerba/nlp/DLNLP_assignment_25/multisource_loss_curves.png")

if __name__ == "__main__":
    main()