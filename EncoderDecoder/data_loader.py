from torchtext.datasets import Multi30k
from torchtext.data.utils import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator
from torch.utils.data import DataLoader

SRC_LANGUAGE = 'de'
TGT_LANGUAGE = 'en'

# Tokenizers
de_tokenizer = get_tokenizer('spacy', language='de_core_news_sm')
en_tokenizer = get_tokenizer('spacy', language='en_core_web_sm')

def build_vocab(dataset, language):
    def yield_tokens():
        for example in dataset:
            yield de_tokenizer(example[0]) if language == SRC_LANGUAGE else en_tokenizer(example[1])
    
    vocab = build_vocab_from_iterator(
        yield_tokens(),
        min_freq=2,
        specials=['<unk>', '<pad>', '<sos>', '<eos>']
    )
    vocab.set_default_index(vocab['<unk>'])
    return vocab

def get_datasets():
    train_iter = Multi30k(split='train', language_pair=(SRC_LANGUAGE, TGT_LANGUAGE))
    valid_iter = Multi30k(split='valid', language_pair=(SRC_LANGUAGE, TGT_LANGUAGE))
    
    # Build vocabularies
    de_vocab = build_vocab(train_iter, SRC_LANGUAGE)
    en_vocab = build_vocab(train_iter, TGT_LANGUAGE)
    
    return train_iter, valid_iter, de_vocab, en_vocab

def collate_fn(batch, de_vocab, en_vocab, device):
    src_batch, tgt_batch = [], []
    for de_text, en_text in batch:
        src_tensor = torch.LongTensor([de_vocab['<sos>']] + 
                      de_vocab(de_tokenizer(de_text)) + 
                      [de_vocab['<eos>']])
        tgt_tensor = torch.LongTensor([en_vocab['<sos>']] + 
                      en_vocab(en_tokenizer(en_text)) + 
                      [en_vocab['<eos>']])
        src_batch.append(src_tensor)
        tgt_batch.append(tgt_tensor)
    
    src_batch = nn.utils.rnn.pad_sequence(src_batch, padding_value=1)  # pad_idx=1
    tgt_batch = nn.utils.rnn.pad_sequence(tgt_batch, padding_value=1)
    
    return src_batch.to(device), tgt_batch.to(device)