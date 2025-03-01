from model import GRUClassifier
from data_loader import SentimentDataset, build_vocab, collate_batch
from config import GRU_CONFIG, TRAIN_CONFIG
from train import GRUTrainer
from torch.utils.data import DataLoader

def main():
    # Load and prepare data
    train_data = SentimentDataset(split='train')
    valid_data = SentimentDataset(split='test')
    vocab = build_vocab(train_data, TRAIN_CONFIG)
    
    # Create data loaders
    train_loader = DataLoader(
        train_data.dataset,
        batch_size=TRAIN_CONFIG['batch_size'],
        shuffle=True,
        collate_fn=lambda x: collate_batch(x, vocab, TRAIN_CONFIG)
    )
    valid_loader = DataLoader(
        valid_data.dataset,
        batch_size=TRAIN_CONFIG['batch_size'],
        collate_fn=lambda x: collate_batch(x, vocab, TRAIN_CONFIG)
    )
    
    # Initialize model
    model = GRUClassifier(
        vocab_size=len(vocab),
        pad_idx=vocab['<pad>'],
        **GRU_CONFIG
    )
    
    # Initialize trainer
    trainer = GRUTrainer(model, train_loader, valid_loader, TRAIN_CONFIG)
    
    # Training loop
    for epoch in range(TRAIN_CONFIG['epochs']):
        train_loss, train_acc = trainer.train_epoch()
        valid_loss, valid_acc = trainer.evaluate()
        
        print(f"\nEpoch {epoch+1}/{TRAIN_CONFIG['epochs']}")
        print(f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc*100:.2f}%")
        print(f"Valid Loss: {valid_loss:.4f} | Valid Acc: {valid_acc*100:.2f}%")
    
    trainer.save_model()

if __name__ == "__main__":
    main()