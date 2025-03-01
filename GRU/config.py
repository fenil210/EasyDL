from torch import nn

# Model Configuration
GRU_CONFIG = {
    'embedding_dim': 300,
    'hidden_dim': 512,
    'output_dim': 1,  # Binary classification
    'n_layers': 2,
    'dropout': 0.3
}

# Training Configuration
TRAIN_CONFIG = {
    'batch_size': 128,
    'epochs': 10,
    'learning_rate': 0.001,
    'optimizer': torch.optim.Adam,
    'loss_fn': nn.BCEWithLogitsLoss(),
    'device': 'cuda' if torch.cuda.is_available() else 'cpu',
    'max_vocab_size': 50000,
    'max_seq_length': 150,
    'min_freq': 2
}