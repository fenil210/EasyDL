from tqdm import tqdm
from torch.nn.utils import clip_grad_norm_
from nltk.translate.bleu_score import corpus_bleu

class Trainer:
    def __init__(self, model, train_loader, valid_loader, config):
        self.model = model
        self.train_loader = train_loader
        self.valid_loader = valid_loader
        self.config = config
        self.optimizer = config['optimizer'](model.parameters(), lr=config['learning_rate'])
        self.loss_fn = config['loss_fn']
        
    def train_epoch(self):
        self.model.train()
        epoch_loss = 0
        
        for src, tgt in tqdm(self.train_loader, desc="Training"):
            self.optimizer.zero_grad()
            output = self.model(src, tgt)
            output = output[1:].view(-1, output.shape[-1])
            tgt = tgt[1:].view(-1)
            loss = self.loss_fn(output, tgt)
            loss.backward()
            clip_grad_norm_(self.model.parameters(), self.config['clip'])
            self.optimizer.step()
            epoch_loss += loss.item()
            
        return epoch_loss / len(self.train_loader)
    
    def evaluate(self):
        self.model.eval()
        epoch_loss = 0
        references = []
        hypotheses = []
        
        with torch.no_grad():
            for src, tgt in self.valid_loader:
                output = self.model(src, tgt, teacher_forcing_ratio=0)
                output = output[1:].view(-1, output.shape[-1])
                tgt = tgt[1:].view(-1)
                loss = self.loss_fn(output, tgt)
                epoch_loss += loss.item()
                
                # For BLEU score
                output_ids = output.argmax(1).view(-1, tgt.shape[0])
                tgt_ids = tgt.view(-1, tgt.shape[0])
                
                for i in range(output_ids.shape[0]):
                    ref = [tgt_ids[i].tolist()]
                    hyp = output_ids[i].tolist()
                    references.append(ref)
                    hypotheses.append(hyp)
                    
        bleu = corpus_bleu(references, hypotheses)
        return epoch_loss / len(self.valid_loader), bleu