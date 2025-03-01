"""
Contains the Preprocessor class which is used for various preprocessing tasks, 
such as file reading, tokenisation, vocabulary creation, etc.
"""

# Import necessary libraries
import json

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
        with open(path, "r") as file:
            content = file.readlines()
        return content
    

    @staticmethod
    def create_parallel_dictionary(text: list[str], save: bool = False) -> list[dict[str, str]]:
        """
        Creates a dictionary of parallel sentences between source and target
        languages from a list of strings.
        """
        # Empty dictionary to store the parallel sentences
        # (from source language to target language)
        translations = []

        for line in text:
            # Get the English and Spanish sentences
            sections = line.split("\t")
            eng_sentence, spa_sentence = sections[0], sections[1]

            # Strip the sentences and lowercase them
            eng_sentence = eng_sentence.strip().lower()
            spa_sentence = spa_sentence.strip().lower()

            # Add the sentences to the dictionary
            translations.append({"eng": eng_sentence, "spa": spa_sentence})

        if save:
            with open("data/translations.json", "w") as file:
                json.dump(translations, file, indent=4)
                file.close()

        return translations
    

    @staticmethod
    def load_dataset(path: str) -> list[dict[str, str]]:
        """
        Loads the dataset from the given path and returns the parallel
        sentences as a list of dictionaries.
        """
        with open(path, "r") as file:
            data = json.load(file)
            file.close()

        return data
