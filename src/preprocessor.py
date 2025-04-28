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
from transformers import T5Tokenizer
import nltk
nltk.download("punkt_tab")
nltk.download("words")
from nltk.tokenize.legality_principle import LegalitySyllableTokenizer
from nltk.tokenize.simple import CharTokenizer
from nltk.corpus import words

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
    def create_tokenised_dataset(translation_dictionary: dict[str, str]) -> list[dict[str, str]]:
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

        return list(map(lambda x: Preprocessor.__create_tokens(x, tokenisers, sos_token="<sos>", eos_token="<eos>"), translation_dictionary))


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
        return list(map(lambda x: Preprocessor.__token_to_index(x, vocabularies), tokenised_dictionaries))
