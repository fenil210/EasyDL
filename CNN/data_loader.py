from torchvision import datasets, transforms
from torch.utils.data import DataLoader

def get_dataloaders(dataset_name: str = 'MNIST', batch_size: int = 64):
    """Create train and test dataloaders for specified dataset."""
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))  # Adjust for different datasets
    ])
    # ADD Datasets as needed
    if dataset_name == 'MNIST':
        train_data = datasets.MNIST(
            root='data', train=True, download=True, transform=transform)
        test_data = datasets.MNIST(
            root='data', train=False, download=True, transform=transform)
    elif dataset_name == 'CIFAR10':
        train_data = datasets.CIFAR10(
            root='data', train=True, download=True, transform=transform)
        test_data = datasets.CIFAR10(
            root='data', train=False, download=True, transform=transform)
    else:
        raise ValueError(f"Unsupported dataset: {dataset_name}")
    
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False)
    
    return train_loader, test_loader