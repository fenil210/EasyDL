import torch
import torch.nn as nn

class LSTMClassifier(nn.Module):
    """Modular LSTM for sequence classification"""
    
    def __init__(self, vocab_size: int, embedding_dim: int, hidden_dim: int,
                 output_dim: int, n_layers: int, dropout: float, pad_idx: int):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=pad_idx)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, num_layers=n_layers,
                           dropout=dropout, bidirectional=False, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, text: torch.Tensor, text_lengths: torch.Tensor) -> torch.Tensor:
        embedded = self.dropout(self.embedding(text))
        packed_embedded = nn.utils.rnn.pack_padded_sequence(
            embedded, text_lengths.cpu(), batch_first=True, enforce_sorted=False)
        packed_output, (hidden, cell) = self.lstm(packed_embedded)
        hidden = self.dropout(hidden[-1, :, :])
        return self.fc(hidden)