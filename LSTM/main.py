from model import LSTMClassifier
from data_loader import get_datasets
from config import LSTM_CONFIG, TRAIN_CONFIG
from train import LSTMTrainer
from torch.utils.data import DataLoader

def main():
    # Get datasets and vocab
    train_iter, test_iter, vocab, collate_fn = get_datasets()
    
    # Create data loaders
    train_loader = DataLoader(list(train_iter), batch_size=TRAIN_CONFIG['batch_size'],
                             shuffle=True, collate_fn=collate_fn)
    test_loader = DataLoader(list(test_iter), batch_size=TRAIN_CONFIG['batch_size'],
                            shuffle=False, collate_fn=collate_fn)
    
    # Initialize model
    model = LSTMClassifier(
        vocab_size=len(vocab),
        pad_idx=vocab['<pad>'],
        **LSTM_CONFIG
    )
    
    # Initialize trainer
    trainer = LSTMTrainer(model, train_loader, test_loader, TRAIN_CONFIG)
    
    # Training loop
    for epoch in range(TRAIN_CONFIG['epochs']):
        train_loss, train_acc = trainer.train_epoch()
        test_loss, test_acc = trainer.evaluate()
        
        print(f"Epoch {epoch+1}/{TRAIN_CONFIG['epochs']}")
        print(f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc*100:.2f}%")
        print(f"Test Loss: {test_loss:.4f} | Test Acc: {test_acc*100:.2f}%")
    
    trainer.save_model()

if __name__ == "__main__":
    main()