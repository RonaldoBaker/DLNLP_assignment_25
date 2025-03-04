# Import dependencies
from functools import partial
from torch.utils.data import DataLoader
from src.preprocessor import Preprocessor
from src.custom_dataset import TranslationDataset, collate_fn

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

    # Define the dataloader
    dataloader = DataLoader(translation_dataset, batch_size=32, shuffle=True, collate_fn=partial(collate_fn, source_vocab=eng_vocab, target_vocab=spa_vocab))
    print("Dataloader created")

    # Model training
    # Define the model
    # Define the loss function
    # Define the optimiser
    # Train the model
    # Evaluate the model


    # Ablation studies on different combinations of embeddings - subword, word and phrase embeddings
    # Train the model with different combinations of embeddings compared to baseline model with just word embeddings
    # Evaluate the model
    # Compare the results

    pass

if __name__ == '__main__':
    main()