from torchtext.datasets import IMDB
from torchtext.data.utils import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator
from torch.utils.data import DataLoader
import torch

tokenizer = get_tokenizer('basic_english')

def yield_tokens(data_iter):
    for _, text in data_iter:
        yield tokenizer(text)

def get_datasets():
    # Load and split dataset
    train_iter, test_iter = IMDB(split=('train', 'test'))
    
    # Build vocabulary
    vocab = build_vocab_from_iterator(yield_tokens(train_iter), 
                                    max_words=config.TRAIN_CONFIG['max_vocab_size'],
                                    specials=['<unk>', '<pad>'])
    vocab.set_default_index(vocab['<unk>'])
    
    # Text processing pipeline
    text_pipeline = lambda x: vocab(tokenizer(x))
    label_pipeline = lambda x: int(x) - 0.5
    
    def collate_batch(batch):
        text_list, label_list, lengths = [], [], []
        for (_label, _text) in batch:
            processed_text = torch.tensor(text_pipeline(_text)[:config.TRAIN_CONFIG['max_seq_length']], 
                              dtype=torch.int64)
            text_list.append(processed_text)
            label_list.append(label_pipeline(_label))
            lengths.append(len(processed_text))
            
        padded_text = nn.utils.rnn.pad_sequence(text_list, padding_value=vocab['<pad>'])
        labels = torch.tensor(label_list, dtype=torch.float32)
        lengths = torch.tensor(lengths, dtype=torch.int64)
        
        return padded_text.T, labels, lengths
    
    return train_iter, test_iter, vocab, collate_batch