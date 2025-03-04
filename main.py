# Import dependencies
import spacy
from src.preprocessor import Preprocessor

def main():
    # DATA PREPROCESSING
    # Read the text file
    lines = Preprocessor.read_text_file("data/spa-eng/spa.txt")
    print("Raw text data read")

    translation_dictionary = Preprocessor.create_parallel_data(text=lines, format="dict", save=False)
    print("Created dictionary of parallel sentences")

    # Create spacy tokenisers
    spa_tokeniser = spacy.load("es_core_news_sm")
    eng_tokeniser = spacy.load("en_core_web_sm")
    print("Loaded spacy tokenisers")

    # Tokenise the data
    tokenised_dataset = Preprocessor.create_tokenised_dataset(eng_tokeniser, spa_tokeniser, translation_dictionary)
    print("Dataset tokenised")

    # Create vocabulary
    eng_vocab, spa_vocab = Preprocessor.build_vocab(tokenised_dataset)
    print("Vocabulary built")

    # Convert the tokenised data to indices

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