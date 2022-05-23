import torch
from torch.nn.utils.rnn import pack_sequence
from torch.utils.data import Dataset


class TextDataset(Dataset):
    def __init__(self, labels, vocab: list[torch.Tensor]):
        """
        A Dataset for the task
        :param labels: corresponding toxicity ratings
        :param vocab: vocabulary with indexed tokens
        """
        self.labels = labels
        self.sentences_embeddings = vocab

    def __getitem__(self, item) -> (int, torch.Tensor):
        return self.sentences_embeddings[item], self.labels[item]

    def __len__(self):
        return len(self.sentences_embeddings)

    def collate_fn(self, batch):
        """
        Technical method to form a batch to feed into recurrent network
        """
        return pack_sequence([torch.tensor(pair[0]) for pair in batch], enforce_sorted=False), torch.tensor(
            [pair[1] for pair in batch])
