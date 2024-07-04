import torch
from torchtext.data.utils import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence

from src.utils import preprocess_file


class SentimentDataset(Dataset):
    def __init__(self, filepath):
        self.texts, self.labels = preprocess_file(filepath)
        self.tokenizer = get_tokenizer("basic_english")
        self.vocab = build_vocab_from_iterator(
            map(self.tokenizer, self.texts), specials=["<unk>"]
        )
        self.vocab.set_default_index(self.vocab["<unk>"])

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.labels[idx] - 1  # adjusting labels to 0-based index
        tokenized_text = torch.tensor(
            self.vocab(self.tokenizer(text)), dtype=torch.long
        )
        return tokenized_text, label

    def collate_batch(self, batch):
        text_list, label_list = [], []
        for _text, _label in batch:
            text_list.append(_text)
            label_list.append(_label)
        text_list = pad_sequence(
            text_list, padding_value=self.vocab["<pad>"], batch_first=True
        )
        label_list = torch.tensor(label_list, dtype=torch.long)
        text_list = text_list.to(torch.device("mps"))
        label_list = label_list.to(torch.device("mps"))
        return text_list, label_list
