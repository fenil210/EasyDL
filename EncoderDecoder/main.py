from model import Encoder, Decoder, Seq2Seq, Attention
from data_loader import get_datasets, collate_fn
from config import ENC_CONFIG, DEC_CONFIG, TRAIN_CONFIG
from train import Trainer

def main():
    # Load dataset and vocabularies
    train_iter, valid_iter, de_vocab, en_vocab = get_datasets()
    
    # Update config with actual vocab sizes
    ENC_CONFIG['input_dim'] = len(de_vocab)
    DEC_CONFIG['output_dim'] = len(en_vocab)
    
    # Initialize model
    attention = Attention(ENC_CONFIG['hid_dim'])
    encoder = Encoder(**ENC_CONFIG)
    decoder = Decoder(attention=attention, **DEC_CONFIG)
    model = Seq2Seq(encoder, decoder, TRAIN_CONFIG['device'])
    
    # Create data loaders
    train_loader = DataLoader(list(train_iter), 
                            batch_size=TRAIN_CONFIG['batch_size'],
                            collate_fn=lambda x: collate_fn(x, de_vocab, en_vocab, TRAIN_CONFIG['device']))
    
    valid_loader = DataLoader(list(valid_iter),
                            batch_size=TRAIN_CONFIG['batch_size'],
                            collate_fn=lambda x: collate_fn(x, de_vocab, en_vocab, TRAIN_CONFIG['device']))
    
    # Initialize trainer
    trainer = Trainer(model, train_loader, valid_loader, TRAIN_CONFIG)
    
    # Training loop
    for epoch in range(TRAIN_CONFIG['epochs']):
        train_loss = trainer.train_epoch()
        valid_loss, bleu = trainer.evaluate()
        
        print(f"Epoch {epoch+1}/{TRAIN_CONFIG['epochs']}")
        print(f"Train Loss: {train_loss:.4f} | Val Loss: {valid_loss:.4f}")
        print(f"BLEU Score: {bleu:.4f}")
    
    torch.save(model.state_dict(), 'seq2seq_multi30k.pth')

if __name__ == "__main__":
    main()