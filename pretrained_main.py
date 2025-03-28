# Import dependencies
import torch
from functools import partial
from torch.utils.data import DataLoader
from torch import nn
from torch import optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from sklearn.model_selection import train_test_split
from transformers import MarianMTModel
from tqdm import tqdm
from nltk.translate.bleu_score import corpus_bleu
from transformers import MarianMTModel
from src.preprocessor import Preprocessor
from src.custom_dataset import TranslationDataset, collate_fn
from src.model import Transformer
from src.model_trainer import TransformerTrainer

# Define the hyperparameters
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

# Set device
if torch.cuda.is_available():
    device_num = 0
    torch.cuda.set_device(device_num)
    device = torch.device(f"cuda:{device_num}")
else:
    device = torch.device("cpu")
print(f"Running on {device}")


def change_pad_index(sequence, old_pad_index, new_pad_index):
    """
    Change the padding index of the sequence.
    """
    sequence = sequence.clone()  # Clone the sequence to avoid modifying the original
    sequence[sequence == old_pad_index] = new_pad_index
    return sequence

def make_src_attention_mask(src, src_pad_index):
    """
    Create a source attention mask for the input sequence.
    The mask will have 1s for the padding tokens and 0s for the actual tokens.
    """
    # Create a boolean mask where True indicates the non-padding tokens that should be attended to
    # and False indicates the padding tokens that should be ignored
    boolean_mask = src != src_pad_index

    # Convert the boolean mask to a float tensor
    # where 1s indicate the tokens to be attended to and 0s indicate the padding tokens
    src_attention_mask = boolean_mask.float()
    return src_attention_mask


def update_model_parameters(model, new_target_vocab_size, new_source_vocab_size):
    # Update the model parameters to match the new vocabulary sizes
    d_model = model.config.d_model # Get the model's hidden dimension

    # Create new embedding layers for the encoder and decoder
    model.model.encoder.embed_tokens = nn.Embedding(new_source_vocab_size, d_model)
    model.model.decoder.embed_tokens = nn.Embedding(new_target_vocab_size, d_model)

    # Update the lm_head to match the new target vocabulary size
    model.lm_head = nn.Linear(d_model, new_target_vocab_size, bias=False)

    # Update the model configuration
    model.config.source_vocab_size = new_source_vocab_size
    model.config.vocab_size = new_target_vocab_size

    return model


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
    translation_dataset = TranslationDataset(indexed_dataset, eng_vocab, spa_vocab, device)
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

    # MODEL INITIALISATION
    # Load the pretrained model
    model_name = "Helsinki-NLP/opus-mt-en-es"
    model = MarianMTModel.from_pretrained(model_name).to(device)
    tuned_model = update_model_parameters(model, len(spa_vocab), len(eng_vocab)).to(device)
    print("Pretrained model loaded and updated")

    # Define the optimiser
    optimiser = optim.Adam(model.parameters(), lr=lr)

    # MarianMT uses a specific index for padding token to ignore during training
    ignore_index = -100
    src_pad_index = eng_vocab["<pad>"]
    tgt_pad_index = spa_vocab["<pad>"]


    train_losses = []
    val_losses = []

    def train_model():
        for i in range(epochs):
            model.train()
            train_loss = 0.0
            for src, tgt in train_loader:
                # Zero the gradients
                optimiser.zero_grad()

                # Make the source attention mask
                src_attention_mask = make_src_attention_mask(src, src_pad_index).to(device)

                # Convert the target sequences to the correct padding index
                tgt = change_pad_index(tgt, tgt_pad_index, ignore_index).to(device)

                # Forward pass
                outputs = model(input_ids=src, attention_mask=src_attention_mask, labels=tgt)
                loss = outputs.loss
                loss.backward()
                optimiser.step()
                train_loss += loss.item()

            train_loss /= len(train_loader)
            train_losses.append(train_loss)

            val_loss = 0.0
            model.eval()
            with torch.no_grad():
                for src, tgt in val_loader:
                    # Make the source attention mask
                    src_attention_mask = make_src_attention_mask(src, src_pad_index).to(device)

                    # Convert the target sequences to the correct padding index
                    tgt = change_pad_index(tgt, tgt_pad_index, ignore_index).to(device)

                    # Forward pass
                    outputs = model(input_ids=src, attention_mask=src_attention_mask, labels=tgt)
                    loss = outputs.loss
                    val_loss += loss.item()

            val_loss /= len(val_loader)
            val_losses.append(val_loss)

            print(f"Epoch {i+1}/{epochs} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")

    def evaluate():

        predictions = []
        references = []

        model.eval()
        with torch.no_grad():
            for src, tgt in test_loader:
                print(src.shape, tgt.shape)
                # Make the source attention mask
                src_attention_mask = make_src_attention_mask(src, src_pad_index).to(device)

                # Forward pass
                predicted_ids = tuned_model.generate(input_ids=src, attention_mask=src_attention_mask, max_length=max_len)
                print(predicted_ids.shape)

                # Convert the predicted IDs to text
                for pred_ids, ref_ids in zip(predicted_ids, tgt):
                    pred_ids = pred_ids[pred_ids != spa_vocab["<pad>"]].tolist()
                    ref_ids = ref_ids[ref_ids != spa_vocab["<pad>"]].tolist()
                
                    # Convert the predicted and reference IDs to text
                    pred_text = spa_vocab.lookup_tokens(pred_ids)
                    ref_text = spa_vocab.lookup_tokens(ref_ids)

                    # Append the predicted and reference texts to the lists
                    predictions.append(pred_text)
                    references.append(ref_text)

        # Calculate BLEU score
        bleu_score = corpus_bleu(references, predictions)
        print(f"BLEU score: {bleu_score:.4f}")
        return bleu_score
    
    evaluate()


if __name__ == "__main__":
    main()
                    