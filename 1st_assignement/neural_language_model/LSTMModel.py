import torch
import torch.nn as nn
from tqdm import tqdm
import numpy as np

class LSTMLanguageModel(nn.Module):
    def __init__(self, embedding_dim, hidden_dim, vocab_size, num_layers=1, dropout=0.5):
        super(LSTMLanguageModel, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        # LSTM layer
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, num_layers, batch_first=True)
        
        # Dropout before the fully connected layer
        self.dropout = nn.Dropout(dropout)
        
        # Linear layer to map from hidden state to vocabulary size
        self.fc = nn.Linear(hidden_dim, vocab_size)
        
    def forward(self, x, hidden):
        # LSTM forward pass
        out, hidden = self.lstm(x, hidden)
        
        # Apply dropout before the linear layer
        out = self.dropout(out)
        
        # Apply linear layer to LSTM output (only last time step's output is relevant)
        out = self.fc(out[:, -1, :])  # We only need the output of the last LSTM time step
        
        return out, hidden
    
    def init_hidden(self, batch_size, device):
        # Initialize hidden state and cell state with zeros
        h0 = torch.zeros(self.num_layers, batch_size, self.hidden_dim).to(device)
        c0 = torch.zeros(self.num_layers, batch_size, self.hidden_dim).to(device)
        return (h0, c0)
    
    def perplexity(self, data_loader, loss_function, device):
        # Set the model to evaluation mode
        self.eval()
        total_loss = 0.0
        total_count = 0
        
        with torch.no_grad():
            for context_embeddings, target_indices in data_loader:
                context_embeddings, target_indices = context_embeddings.to(device), target_indices.to(device)
                batch_size = context_embeddings.size(0)
                
                hidden = self.init_hidden(batch_size, device)
                output, hidden = self(context_embeddings, hidden)
                
                loss = loss_function(output, target_indices)
                total_loss += loss.item() * batch_size
                total_count += batch_size
        
        # Calculate perplexity
        average_loss = total_loss / total_count
        perplexity = np.exp(average_loss)
        
        return perplexity
    
    def predict(self, context_embeddings, device):
            # Set the model to evaluation mode
            self.eval()
            
            # Prepare context embeddings
            # context_embeddings = context_embeddings.unsqueeze(0).to(device)  # Add batch dimension
            context_embeddings = context_embeddings.to(device)  # Add batch dimension
            
            with torch.no_grad():
                batch_size = context_embeddings.size(0)
                hidden = self.init_hidden(batch_size, device)
                output, hidden = self(context_embeddings, hidden)
            
            # Get the predicted word index
            predicted_word_idx = output.argmax(dim=1)[0].item()
            
            return predicted_word_idx

def train_and_evaluate(model, train_loader, val_loader, optimizer, loss_function, num_epochs, device):
    model.to(device)
    
    for epoch in tqdm(range(num_epochs), desc='Training'):
        model.train()
        total_train_loss = 0
        
        for context_embeddings, target_indices in train_loader:
            context_embeddings, target_indices = context_embeddings.to(device), target_indices.to(device)
            batch_size = context_embeddings.size(0)
            
            hidden = model.init_hidden(batch_size, device)
            
            optimizer.zero_grad()
            output, hidden = model(context_embeddings, hidden)
            
            loss = loss_function(output, target_indices)
            loss.backward()
            optimizer.step()
            
            total_train_loss += loss.item()
        
        average_train_loss = total_train_loss / len(train_loader)
        print(f'Epoch {epoch+1}, Training Loss: {average_train_loss:.4f}')
        
        # Evaluation phase
        model.eval()
        total_val_loss = 0
        
        with torch.no_grad():
            for context_embeddings, target_indices in val_loader:
                context_embeddings, target_indices = context_embeddings.to(device), target_indices.to(device)
                batch_size = context_embeddings.size(0)
                
                hidden = model.init_hidden(batch_size, device)
                output, hidden = model(context_embeddings, hidden)
                
                loss = loss_function(output, target_indices)
                total_val_loss += loss.item()
        
        average_val_loss = total_val_loss / len(val_loader)
        perplexity = np.exp(average_val_loss)
        print(f'Epoch {epoch+1}, Validation Loss: {average_val_loss:.4f}, Perplexity: {perplexity:.4f}')


