from torch import nn

# Model Configuration
ENC_CONFIG = {
    'input_dim': 10000,  # Will be set from vocab
    'emb_dim': 256,
    'hid_dim': 512,
    'dropout': 0.5
}

DEC_CONFIG = {
    'output_dim': 10000,  # Will be set from vocab
    'emb_dim': 256,
    'hid_dim': 512,
    'dropout': 0.5
}

# Training Configuration
TRAIN_CONFIG = {
    'batch_size': 128,
    'epochs': 20,
    'learning_rate': 0.001,
    'teacher_forcing_ratio': 0.5,
    'clip': 1.0,
    'optimizer': torch.optim.Adam,
    'loss_fn': nn.CrossEntropyLoss(ignore_index=1),  # Ignore padding
    'device': 'cuda' if torch.cuda.is_available() else 'cpu'
}