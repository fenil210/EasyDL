from tqdm import tqdm
import torch

class LSTMTrainer:
    """Training and evaluation module for LSTM"""
    
    def __init__(self, model, train_loader, test_loader, config):
        self.model = model.to(config['device'])
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.config = config
        self.optimizer = config['optimizer'](model.parameters(), lr=config['learning_rate'])
        self.loss_fn = config['loss_fn']
        
    def train_epoch(self):
        self.model.train()
        total_loss, total_acc = 0, 0
        
        for text, labels, lengths in tqdm(self.train_loader, desc="Training"):
            text, lengths = text.to(self.config['device']), lengths.to(self.config['device'])
            labels = labels.to(self.config['device'])
            
            self.optimizer.zero_grad()
            predictions = self.model(text, lengths).squeeze(1)
            loss = self.loss_fn(predictions, labels)
            acc = ((torch.sigmoid(predictions) > 0.5).float() == (labels > 0)).float().mean()
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1)
            self.optimizer.step()
            
            total_loss += loss.item()
            total_acc += acc.item()
            
        return total_loss / len(self.train_loader), total_acc / len(self.train_loader)
    
    def evaluate(self):
        self.model.eval()
        total_loss, total_acc = 0, 0
        
        with torch.no_grad():
            for text, labels, lengths in self.test_loader:
                text, lengths = text.to(self.config['device']), lengths.to(self.config['device'])
                labels = labels.to(self.config['device'])
                
                predictions = self.model(text, lengths).squeeze(1)
                loss = self.loss_fn(predictions, labels)
                acc = ((torch.sigmoid(predictions) > 0.5).float() == (labels > 0)).float().mean()
                
                total_loss += loss.item()
                total_acc += acc.item()
                
        return total_loss / len(self.test_loader), total_acc / len(self.test_loader)
    
    def save_model(self, path='lstm_classifier.pth'):
        torch.save(self.model.state_dict(), path)