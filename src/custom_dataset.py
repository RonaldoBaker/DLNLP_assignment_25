# Import dependencies
import torch
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence


class TranslationDataset(Dataset):
    """
    A custom dataset class for the translation task.
    """
    def __init__(self, indexed_dataset, source_vocab, target_vocab, device):
        """
        Initialises the dataset with the indexed dataset and the source and target vocabularies.
        """
        self.indexed_dataset = indexed_dataset
        self.source_vocab = source_vocab
        self.target_vocab = target_vocab
        self.device = device


    def __len__(self):
        """
        Returns the length of the dataset.
        """
        return len(self.indexed_dataset)
    

    def __getitem__(self, idx):
        """
        Returns the indexed source and target sentences at the given index.
        """
        dictionary = self.indexed_dataset[idx]
        eng_tensor = torch.tensor(dictionary['eng_ids'], dtype=torch.long, device=self.device)
        spa_tensor = torch.tensor(dictionary['spa_ids'], dtype=torch.long, device=self.device)
        return eng_tensor, spa_tensor


class MultiTokenisationDataset(Dataset):
    def __init__(self, indexed_dataset, device):
        self.indexed_data = indexed_dataset
        self.device = device

    def __len__(self):
        return len(self.indexed_data)

    def __getitem__(self, idx):
        dictionary = self.indexed_data[idx]
        src_dict = {key: torch.tensor(value, dtype=torch.long, device=self.device) for key, value in dictionary.items() 
                    if key.endswith("_ids") and key.startswith("src_")}
        tgt_tensor = torch.tensor(dictionary['tgt_word_ids'], dtype=torch.long, device=self.device)
        return src_dict, tgt_tensor


def collate_fn(batch, source_vocab, target_vocab):
    eng_tensors, spa_tensors = zip(*batch)
    eng_tensors = pad_sequence(eng_tensors, padding_value=source_vocab["<pad>"], batch_first=True)
    spa_tensors = pad_sequence(spa_tensors, padding_value=target_vocab["<pad>"], batch_first=True)
    return eng_tensors, spa_tensors


def collate_fn_multitokenisation(batch, source_vocab, target_vocab):
    dicts, tensors = zip(*batch)  # Unpack dictionaries and tensors from the batch

    # Pad the dictionaries
    padded_dicts = {key: pad_sequence([d[key] for d in dicts], padding_value=source_vocab["<pad>"], batch_first=True) for key in dicts[0].keys()}

    # Pad the individual tensors
    padded_tensors = pad_sequence(tensors, padding_value=target_vocab["<pad>"], batch_first=True)

    return padded_dicts, padded_tensors