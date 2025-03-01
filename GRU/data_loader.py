from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset
from torchtext.data.utils import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator
import torch

tokenizer = get_tokenizer('basic_english')

class SentimentDataset(Dataset):
    """Dataset class for Sentiment140"""
    
    def __init__(self, split='train'):
        self.dataset = load_dataset('sentiment140', split=split)
        self.texts = [example['text'] for example in self.dataset]
        self.labels = [example['sentiment'] for example in self.dataset]

def build_vocab(dataset, config):
    def yield_tokens():
        for text in dataset.texts:
            yield tokenizer(text)[:config['max_seq_length']]
            
    vocab = build_vocab_from_iterator(
        yield_tokens(),
        max_tokens=config['max_vocab_size'],
        min_freq=config['min_freq'],
        specials=['<unk>', '<pad>']
    )
    vocab.set_default_index(vocab['<unk>'])
    return vocab

def collate_batch(batch, vocab, config):
    text_list, label_list, lengths = [], [], []
    for example in batch:
        text = example['text']
        tokens = tokenizer(text)[:config['max_seq_length']]
        processed_text = torch.tensor([vocab[token] for token in tokens], dtype=torch.int64)
        text_list.append(processed_text)
        label_list.append(example['sentiment'] / 4.0)  # Convert to 0-1 range
        lengths.append(len(processed_text))
    
    padded_text = nn.utils.rnn.pad_sequence(text_list, padding_value=vocab['<pad>'])
    labels = torch.tensor(label_list, dtype=torch.float32)
    lengths = torch.tensor(lengths, dtype=torch.int64)
    
    return padded_text.T, labels, lengths