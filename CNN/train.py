import torch
from tqdm import tqdm

class Trainer:
    """Modular training class for CNN models."""
    
    def __init__(self, model, train_loader, test_loader, config):
        self.model = model.to(config['device'])
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.config = config
        self.optimizer = config['optimizer'](model.parameters(), lr=config['learning_rate'])
        self.loss_fn = config['loss_fn']
        
    def train_epoch(self):
        self.model.train()
        total_loss = 0
        
        for data, targets in tqdm(self.train_loader, desc="Training"):
            data = data.to(self.config['device'])
            targets = targets.to(self.config['device'])
            
            self.optimizer.zero_grad()
            outputs = self.model(data)
            loss = self.loss_fn(outputs, targets)
            loss.backward()
            self.optimizer.step()
            
            total_loss += loss.item()
            
        return total_loss / len(self.train_loader)
    
    def evaluate(self):
        self.model.eval()
        correct = 0
        total = 0
        
        with torch.no_grad():
            for data, targets in self.test_loader:
                data = data.to(self.config['device'])
                targets = targets.to(self.config['device'])
                
                outputs = self.model(data)
                _, predicted = torch.max(outputs.data, 1)
                total += targets.size(0)
                correct += (predicted == targets).sum().item()
                
        return correct / total
    
    def save_model(self, path='model.pth'):
        torch.save(self.model.state_dict(), path)