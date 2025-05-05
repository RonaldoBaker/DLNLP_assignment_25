"""
Contains the Preprocessor class which is used for various preprocessing tasks, 
such as file reading, tokenisation, vocabulary creation, etc.
"""

# Import necessary libraries
import os
import sys
import json
import string
import spacy
import torchtext
torchtext.disable_torchtext_deprecation_warning()
from torchtext.vocab import Vocab, build_vocab_from_iterator  # noqa: E402
from transformers import T5Tokenizer
import nltk
nltk.download("punkt_tab")
nltk.download("words")
from nltk.tokenize.legality_principle import LegalitySyllableTokenizer
from nltk.tokenize.simple import CharTokenizer
from nltk.corpus import words
from tqdm import tqdm

# Append project root to sys.path
project_root = os.path.join(os.path.dirname(__file__), "..")
if project_root not in sys.path:
    sys.path.append(project_root)

from src.lattice import Lattice

class Preprocessor:
    """
    The Preprocessor class contains various methods for preprocessing text data.
    """
    # Tokenisation method to vocabulary name mapping
    # when building the vocabulary and changing tokens into integers
    token_to_vocab_map = {
        "src_word_tokens": "src_word_vocab",
        "tgt_word_tokens": "tgt_word_vocab",
        "src_subword_tokens": "src_subword_vocab",
        "src_syllable_tokens": "src_syllable_vocab",
        "src_char_tokens": "src_char_vocab"
    }

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
                translations.append({"src": eng_sentence, "tgt": spa_sentence})
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
                      tokenisers: dict,
                      sos_token: str,
                      eos_token: str,
                      remove_underscore: bool,
                      max_length: int = 100
                      ) -> dict[str, list[str]]:
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
        # Tokenise the text using the tokenisers from the dictionary
        # Word tokenisation
        src_word_tokens = [token.text for token in tokenisers["src_word_tokeniser"].tokenizer(pair["src"])][:max_length]
        tgt_word_tokens = [token.text for token in tokenisers["tgt_word_tokeniser"].tokenizer(pair["tgt"])][:max_length]

        # Subword tokenisation
        src_subword_tokens = tokenisers["src_subword_tokeniser"].tokenize(pair["src"])[:max_length]
        if remove_underscore:
            src_subword_tokens = [token[1:] for token in src_subword_tokens if token.startswith("▁")]
            src_subword_tokens = [token for token in src_subword_tokens if len(token) > 0] # Remove empty tokens

        # Syllable tokenisation
        src_syllable_tokens = [tokenisers["src_syllable_tokeniser"].tokenize(word_token) for word_token in src_word_tokens]
        # Returns a list of lists so convert to one list
        src_syllable_tokens = [item for sublist in src_syllable_tokens for item in sublist][:max_length]

        # Character tokenisation
        """
        Max length is set to max_length - 2 to account for the start and end of sentence tokens
        because char tokens are more likely to exceed the max length and cutting it off at max length
        would still result in a list that is too long
        """
        src_char_tokens = tokenisers["src_char_tokeniser"].tokenize(pair["src"])[:max_length-2]

        # Add the start of sentence and end of sentence tokens
        tokenisations = {
            "src_word_tokens": src_word_tokens,
            "tgt_word_tokens": tgt_word_tokens,
            "src_subword_tokens": src_subword_tokens,
            "src_syllable_tokens": src_syllable_tokens,
            "src_char_tokens": src_char_tokens
            }

        for key in tokenisations.keys():
            tokenisations[key] = [sos_token] + tokenisations[key] + [eos_token]

        pair.update(tokenisations)
        return pair


    @staticmethod
    def create_tokenised_dataset(translation_dictionary: dict[str, str], remove_underscore: bool = True) -> list[dict[str, str]]:
        """
        Wraps the __create tokens function into a lambda function
        and tokenises the parallel sentences in the given dictionary

        Args:
            - translation_dictionary (dict[str, str]): The dictionary containing the parallel sentences

        Returns:
            - (list[dict[str, str]]): The tokenised parallel sentences as a list of dictionaries
        """
         # Create spacy tokenisers for word tokenisation
        src_word_tokeniser = spacy.load("es_core_news_sm")
        tgt_word_tokeniser = spacy.load("en_core_web_sm")

        # Create pretrained huggingface tokeniser for subword tokenisation
        src_subword_tokeniser = T5Tokenizer.from_pretrained("t5-base", legacy=False)

        # Create nltk tokenisers for syllable and character tokenization
        src_syllable_tokeniser = LegalitySyllableTokenizer(words.words())
        src_char_tokeniser = CharTokenizer()

        tokenisers = {"src_word_tokeniser": src_word_tokeniser,
                      "tgt_word_tokeniser": tgt_word_tokeniser,
                      "src_subword_tokeniser": src_subword_tokeniser,
                      "src_syllable_tokeniser": src_syllable_tokeniser,
                      "src_char_tokeniser": src_char_tokeniser
                      }

        return list(map(lambda x: Preprocessor.__create_tokens(x, tokenisers, sos_token="<sos>", eos_token="<eos>", remove_underscore=remove_underscore),
                        tqdm(translation_dictionary, desc="Tokenising data", unit="dictionary", leave=True)))


    @staticmethod
    def build_vocabularies(tokenised_dictionaries: list[dict[str, str]]) -> dict[str, Vocab]:
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

        vocabularies = {} # Dictionary to store the vocabularies

        for token_type, vocab_name in Preprocessor.token_to_vocab_map.items():
            tokens = (dictionary[token_type] for dictionary in tokenised_dictionaries)
            vocab = build_vocab_from_iterator(tokens, specials=special_tokens, min_freq=2)
            vocab.set_default_index(vocab["<unk>"])
            vocabularies[vocab_name] = vocab

        return vocabularies


    @staticmethod
    def __token_to_index(tokenised_dictionary: dict[str, str], vocabularies: dict[str, Vocab]):
        """
        Maps the tokens to their corresponding indices using the vocabulary.

        Args:
            - data (dict[str, str]): The dictionary containing the tokens
            - eng_vocab (Vocab): The English vocabulary
            - spa_vocab (Vocab): The Spanish vocabulary
        
        Returns:
            - (dict[str, str]): The uypdated dictionary containing the indices
        """

        # Using class mapping to access in-built numericalization methods from each Vocab object
        # in order to convert the tokens to indices
        indices_to_add = {}

        for token_type, vocab_name in Preprocessor.token_to_vocab_map.items():
            indices = vocabularies[vocab_name].lookup_indices(tokenised_dictionary[token_type])
            indices_to_add[token_type.replace("_tokens", "_ids")] = indices

        tokenised_dictionary.update(indices_to_add)
        return tokenised_dictionary


    @staticmethod
    def numericalise(tokenised_dictionaries: list[dict[str, str]], vocabularies: dict[str, Vocab]) -> list[dict]:
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
        return list(map(lambda x: Preprocessor.__token_to_index(x, vocabularies),
                        tqdm(tokenised_dictionaries, desc="Numericalising tokenised data", unit="dictionary", leave=True)))
    

    @staticmethod
    def __generate_lpes(dictionary: dict[str, str], fine_grain_tokenisation: str):
        """
        Generates the lattice positional encodings for the given dictionary.

        Arg(s):
            - dictionary (dict[str, str]): The dictionary containing the tokenised data
            - device (str): The device to use for the tensors
            - fine_grain_tokenisation (str): The fine-grain tokenisation to use

        Returns:
            - (dict[str, torch.Tensor]): The dictionary containing the lattice positional encodings
        """
        # Create a lattice object - creates the lattice inherently
        lattice = Lattice(dictionary["src_word_tokens"], dictionary["src_" + fine_grain_tokenisation + "_tokens"], type=fine_grain_tokenisation)

        if lattice.graph is None: # Threw an error, tokenisations cannot be aligned
            lpes_to_add = {
                "src_word_lpes": None,
            }

            dictionary.update(lpes_to_add)

            return dictionary

        else:
            # Get the positional encodings for the word and fine-grain tokens
            word_lpes, fine_grain_lpes = lattice.get_lattice_positional_encodings()

            # Add the positional encodings to the dictionary
            lpes_to_add = {
                "src_word_lpes": word_lpes,
                "src_" + fine_grain_tokenisation + "_lpes": fine_grain_lpes
            }

            # Update the dictionary with the lattice positional encodings
            dictionary.update(lpes_to_add)

            return dictionary


    @staticmethod
    def lattice_building(dictionaries: list[dict[str, str]], fine_grain_tokenisation = str) -> list[dict]:
        """
        Creates the lattice positional encodings for the given dictionaries.

        Args:
            - dictionaries (list[dict[str, str]]): The dictionaries containing the tokenised data
            - device (str): The device to use for the tensors

        Returns:
            - (list[dict[str, torch.Tensor]]): The dictionaries containing the lattice positional encodings
        """
        lattice_dicts = list(map(lambda x: Preprocessor.__generate_lpes(x, fine_grain_tokenisation),
                        tqdm(dictionaries, desc="Generating lattice positional encodings", unit="dictionary", leave=True)))
        
        # Discard any dictionaries where tokenisations could not be aligned
        return [dictionary for dictionary in lattice_dicts if dictionary["src_word_lpes"] is not None]      
