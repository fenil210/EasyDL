from torch import nn

# Model Configuration
LSTM_CONFIG = {
    'embedding_dim': 100,
    'hidden_dim': 256,
    'output_dim': 1,  # Binary classification
    'n_layers': 2,
    'dropout': 0.5
}

# Training Configuration
TRAIN_CONFIG = {
    'batch_size': 64,
    'epochs': 5,
    'learning_rate': 0.001,
    'optimizer': torch.optim.Adam,
    'loss_fn': nn.BCEWithLogitsLoss(),
    'device': 'cuda' if torch.cuda.is_available() else 'cpu',
    'max_vocab_size': 25000,
    'max_seq_length': 500
}