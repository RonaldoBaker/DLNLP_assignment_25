# Import dependencies
import torch
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence


class TokenDataset(Dataset):
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
        eng_tensor = torch.tensor(dictionary['src_word_ids'], dtype=torch.long, device=self.device)
        spa_tensor = torch.tensor(dictionary['tgt_word_ids'], dtype=torch.long, device=self.device)
        return eng_tensor, spa_tensor


class MultiTokenDataset(Dataset):
    def __init__(self, indexed_dataset: dict[str, list[str]], chosen_tokenisations: list[str], device: str, embedding_dim: int = 512, fusion_type: str = "single"):
        self.indexed_data = indexed_dataset
        self.chosen_tokenisations = chosen_tokenisations
        self.device = device
        self.embedding_dim = embedding_dim
        self.fusion_type = fusion_type

    def __len__(self):
        return len(self.indexed_data)

    def __getitem__(self, idx):
        dictionary = self.indexed_data[idx]

        # Take the tokenisations that are in the chosen tokenisations
        src_dict = {key: torch.tensor(value, dtype=torch.long, device=self.device) for key, value in dictionary.items() 
                    if key in self.chosen_tokenisations}
        tgt_tensor = torch.tensor(dictionary['tgt_word_ids'], dtype=torch.long, device=self.device)

        if self.fusion_type == "lattice":
            # Get the lattice positional encodings for fusion
            src_lpes = {key: torch.tensor(value, dtype=torch.float32, device=self.device)
                        for key, value in dictionary.items() if key.endswith("_lpes")}

            src_dict.update(src_lpes)

        return src_dict, tgt_tensor


def collate_fn(batch, source_vocab, target_vocab):
    eng_tensors, spa_tensors = zip(*batch)
    eng_tensors = pad_sequence(eng_tensors, padding_value=source_vocab["<pad>"], batch_first=True)
    spa_tensors = pad_sequence(spa_tensors, padding_value=target_vocab["<pad>"], batch_first=True)
    return eng_tensors, spa_tensors


def collate_fn_multitokenisation(batch, source_vocab, target_vocab, embedding_size: int = 512):
    dicts, tensors = zip(*batch)

    # Pad the dictionaries
    padded_dicts = {key: pad_sequence([d[key] for d in dicts], padding_value=source_vocab["<pad>"], batch_first=True)
                    for key in dicts[0].keys() if not key.endswith("_lpes")}
    
    # Pad any lattice positional encodings and update to the padded dictionaries
    try:
        lpes = {key: pad_sequence([d[key] for d in dicts], padding_value=0, batch_first=True).unsqueeze(2).expand(-1, -1, embedding_size)
                for key in dicts[0].keys() if key.endswith("_lpes")}
        padded_dicts.update(lpes)
    except KeyError:
        pass

    # Pad the individual tensors
    padded_tensors = pad_sequence(tensors, padding_value=target_vocab["<pad>"], batch_first=True)

    return padded_dicts, padded_tensors
    # if lpes is None:
    #     # If no lattice positional encodings, return only the padded dictionaries and tensors
    #     return padded_dicts, padded_tensors
    # else:
    #     # Pad the lattice positional encodings and expand to embedding dimension
    #     padded_lpes = {key: pad_sequence([l[key] for l in lpes],
    #                                     padding_value=0, batch_first=True).unsqueeze(2).expand(-1, -1, embedding_size)
    #                                     for key in lpes[0].keys()}
    #     return padded_dicts, padded_lpes, padded_tensors