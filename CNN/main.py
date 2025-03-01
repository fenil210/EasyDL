from EasyDL.CNN.models import CNN
from EasyDL.CNN.data_loader import get_dataloaders
from EasyDL.CNN.config import CNN_CONFIG, TRAIN_CONFIG
from EasyDL.CNN.train import Trainer

def main():
    # Initialize model
    model = CNN(
        in_channels=1,  # MNIST has 1 channel
        num_classes=10,
        layer_config=CNN_CONFIG
    )
    
    # Get data loaders
    train_loader, test_loader = get_dataloaders(
        dataset_name='MNIST',
        batch_size=TRAIN_CONFIG['batch_size']
    )
    
    # Initialize trainer
    trainer = Trainer(model, train_loader, test_loader, TRAIN_CONFIG)
    
    # Training loop
    for epoch in range(TRAIN_CONFIG['epochs']):
        train_loss = trainer.train_epoch()
        test_acc = trainer.evaluate()
        print(f"Epoch {epoch+1}/{TRAIN_CONFIG['epochs']}")
        print(f"Train Loss: {train_loss:.4f} | Test Accuracy: {test_acc:.4f}")
    
    trainer.save_model()

if __name__ == "__main__":
    main()