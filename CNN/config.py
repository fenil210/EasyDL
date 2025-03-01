import torch.optim as optim

# CNN Architecture Configuration
CNN_CONFIG = [
    {
        'out_channels': 32,
        'kernel_size': 3,
        'stride': 1,
        'padding': 1,
        'activation': 'relu',
        'pool': {'kernel_size': 2, 'stride': 2}
    },
    {
        'out_channels': 64,
        'kernel_size': 3,
        'stride': 1,
        'padding': 1,
        'activation': 'relu',
        'pool': {'kernel_size': 2, 'stride': 2}
    }
]

# Training Configuration
TRAIN_CONFIG = {
    'batch_size': 64,
    'epochs': 10,
    'learning_rate': 0.001,
    'optimizer': optim.Adam,
    'loss_fn': nn.CrossEntropyLoss(),
    'device': 'cuda' if torch.cuda.is_available() else 'cpu'
}