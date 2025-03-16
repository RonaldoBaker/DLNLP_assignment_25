"""
Contains the Preprocessor class which is used for various preprocessing tasks, 
such as file reading, tokenisation, vocabulary creation, etc.
"""

# Import necessary libraries
import json
import string
import spacy
import torchtext
torchtext.disable_torchtext_deprecation_warning()
from torchtext.vocab import Vocab, build_vocab_from_iterator  # noqa: E402

class Preprocessor:
    """
    The Preprocessor class contains various methods for preprocessing text data.
    """
    @staticmethod
    def read_text_file(path: str) -> list[str]:
        """
        Reads a text file and returns the content as a list of strings,
        each line being one element in the list.

        Args:
            - path (str): The path to the text file
        
        Returns:
            - content (list[str]): The content of the text file as a list of strings
        """
        with open(file=path, mode="r", encoding="utf-8") as file:
            content = file.readlines()
        return content
    

    @staticmethod
    def create_parallel_data(text: list[str], format: str, save: bool = False
                             ) -> list[dict[str, str]]:
        # TODO: Come back to this and see if I still need the tuple version so that I can
        # remove it and fix the return type
        """
        Creates a dictionary of parallel sentences between source and target
        languages from a list of strings.

        Args:
            - text (list[str]): The list of strings containing the parallel sentences
            - format (str): The format in which to return the parallel sentences
                            (either "dict" or "tuple")
            - save (bool): Whether to save the parallel sentences to a JSON file or not

        Returns:
            - (list[dict[str, str]]): The parallel sentences as a list of dictionaries or tuples
        """
        # Empty dictionary to store the parallel sentences
        # (from source language to target language)
        translations = []

        for line in text:
            # Get the English and Spanish sentences
            sections = line.split("\t")
            eng_sentence, spa_sentence = sections[0], sections[1]

            # Remove punctuation (including inverted question and exclamation marks for Spanish)
            all_punctuations = string.punctuation + "¡¿"
            eng_sentence = eng_sentence.translate(str.maketrans("","", all_punctuations))
            spa_sentence = spa_sentence.translate(str.maketrans("","", all_punctuations))

            # Strip the sentences and lowercase them
            eng_sentence = eng_sentence.strip().lower()
            spa_sentence = spa_sentence.strip().lower()

            # Add the sentences to the dictionary
            if format == "dict":
                translations.append({"eng": eng_sentence, "spa": spa_sentence})
            elif format == "tuple":
                translations.append((eng_sentence, spa_sentence))

        if save:
            if format == "dict":
                with open("data/translations_dict.json", "w") as file:
                    json.dump(translations, file, indent=4)
                    file.close()
            elif format == "tuple":
                with open("data/translations_tuples.json", "w") as file:
                    json.dump([list(item) for item in translations], file, indent=4)

        return translations
    

    @staticmethod
    def load_dataset(path: str) -> list[dict[str, str]]:
        # TODO: check if I am removing the tuple version of this method
        """
        Loads the dataset from the given path and returns the parallel
        sentences as a list of dictionaries.

        Args:
            - path (str): The path to the dataset file

        Returns:
            - (list[dict[str, str]]): The parallel sentences as a list of dictionaries or tuples
        """
        with open(path, "r") as file:
            data = json.load(file)
            file.close()

        return data


    @staticmethod
    def __create_tokens(pair: dict[str, str],
                      eng_tokeniser: spacy.language.Language,
                      spa_tokeniser: spacy.language.Language,
                      sos_token: str,
                      eos_token: str,
                      max_length: int = 100
                      ) -> dict[str, str]:
        """
        Tokenises the sentences in the given pair of parallel sentences
        and returns the tokenised sentences as a dictionary.

        Args:
            - pair (dict[str, str]): The pair of parallel sentences
            - eng_tokeniser (spacy.langugage.Language): The English tokeniser
            - spa_tokeniser (spacy.langugage.Language): The Spanish tokeniser
            - sos_token (str): The start of sentence token
            - eos_token (str): The end of sentence token
            - max_length (int): The maximum length of the tokenised sentences

        Returns:
            - (dict[str, str]): The tokenised sentences as a dictionary
        """
        # Tokenise the text
        eng_tokens = [token.text for token in eng_tokeniser.tokenizer(pair["eng"])][:max_length]
        spa_tokens = [token.text for token in spa_tokeniser.tokenizer(pair["spa"])][:max_length]

        # Add the start of sentence and end of sentence tokens
        eng_tokens = [sos_token] + eng_tokens + [eos_token]
        spa_tokens = [sos_token] + spa_tokens + [eos_token]

        pair.update({"eng_tokens": eng_tokens, "spa_tokens": spa_tokens})
        return pair
    

    @staticmethod
    def create_tokenised_dataset(translation_dictionary: dict[str, str]) -> list[dict[str, str]]:
        """
        Wraps the __create tokens function into a lambda function
        and tokenises the parallel sentences in the given dictionary

        Args:
            - translation_dictionary (dict[str, str]): The dictionary containing the parallel sentences

        Returns:
            - (list[dict[str, str]]): The tokenised parallel sentences as a list of dictionaries
        """
         # Create spacy tokenisers
        spa_tokeniser = spacy.load("es_core_news_sm")
        eng_tokeniser = spacy.load("en_core_web_sm")

        return list(map(lambda x: Preprocessor.__create_tokens(x, eng_tokeniser, spa_tokeniser, sos_token="<sos>", eos_token="<eos>"),translation_dictionary))
        

    @staticmethod
    def build_vocab(tokenised_data: list[dict[str, str]]) -> tuple[Vocab, Vocab]:
        """
        Builds the vocabulary for the source and target languages
        from the tokenised data and returns the vocabulary of each language.

        Args:
            - tokenised_data (list[dict[str, str]]): The tokenised data

        Returns:
            - (tuple[Vocab, Vocab]): The vocabulary for the source and target languages
        """
        # Define special tokens
        special_tokens = ["<sos>", "<eos>", "<unk>", "<pad>"]

        eng_tokens = [parallel_dict["eng_tokens"] for parallel_dict in tokenised_data]
        spa_tokens = [parallel_dict["spa_tokens"] for parallel_dict in tokenised_data]

        # Build the vocabulary
        eng_vocab = build_vocab_from_iterator(eng_tokens, specials=special_tokens, min_freq=2)
        spa_vocab = build_vocab_from_iterator(spa_tokens, specials=special_tokens, min_freq=2)

        # Set unknown token index as default index
        eng_vocab.set_default_index(eng_vocab["<unk>"])
        spa_vocab.set_default_index(spa_vocab["<unk>"])

        return eng_vocab, spa_vocab


    @staticmethod
    def __token_to_index(data: dict[str, str], eng_vocab: Vocab, spa_vocab: Vocab):
        """
        Maps the tokens to their corresponding indices using the vocabulary.

        Args:
            - data (dict[str, str]): The dictionary containing the tokens
            - eng_vocab (Vocab): The English vocabulary
            - spa_vocab (Vocab): The Spanish vocabulary
        
        Returns:
            - (dict[str, str]): The uypdated dictionary containing the indices
        """
        # Use in-built numericalization methods to convert tokens to indices
        eng_ids = eng_vocab.lookup_indices(data["eng_tokens"])
        spa_ids = spa_vocab.lookup_indices(data["spa_tokens"])

        # Update the dictionary with the indices
        data.update({"eng_ids": eng_ids, "spa_ids": spa_ids})

        return data


    @staticmethod
    def numericalise(tokenised_data: list[dict[str, str]], eng_vocab: Vocab, spa_vocab: Vocab) -> list[dict]:
        """
        Wraps the __token_to_index function into a lambda function
        and converts the tokenised data to indices using the vocabulary.

        Args:
            - tokenised_data (list[dict[str, str]]): The tokenised data
            - eng_vocab (Vocab): The English vocabulary
            - spa_vocab (Vocab): The Spanish vocabulary

        Returns:
            - (list[dict]): The tokenised data as a list of dictionaries containing the indices
        """
        return list(map(lambda x: Preprocessor.__token_to_index(x, eng_vocab, spa_vocab), tokenised_data))
