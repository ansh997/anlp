import torch
import numpy as np
from tqdm import tqdm
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

# should be updated
# Basic medium blog - https://medium.com/@roshmitadey/understanding-language-modeling-from-n-grams-to-transformer-based-neural-models-d2bdf1532c6d

# Check if GPU is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class NNLM(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, num_hidden_layers=2, context_size=5,
                # device='cpu'
                ):
        super(NNLM, self).__init__()
        # Ensure parameters are integers
        assert isinstance(context_size, int), f"Expected context_size to be int but got {type(context_size)}"
        assert isinstance(embedding_dim, int), f"Expected embedding_dim to be int but got {type(embedding_dim)}"
        assert isinstance(hidden_dim, int), f"Expected hidden_dim to be int but got {type(hidden_dim)}"
        
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.context_size = context_size
        
        self.input_dim = context_size * embedding_dim
        
        print(f"context_size: {context_size}, embedding_dim: {embedding_dim}, hidden_dim: {hidden_dim}")
        print(f'{device=}')


        
        # Hidden layers
        # TODO: make it variable
        self.hidden1 = nn.Linear(self.input_dim, hidden_dim)
        self.hidden2 = nn.Linear(hidden_dim, vocab_size)
        
        # Activation functions
        self.relu = nn.ReLU()
        self.softmax = nn.LogSoftmax(dim=1)
    
    
    def forward(self, inputs):
        flattened_embeds = inputs.view(inputs.shape[0], -1)
        hidden1_out = self.relu(self.hidden1(flattened_embeds))
        hidden2_out = self.hidden2(hidden1_out)
        output = self.softmax(hidden2_out)
        return output
    
    def train_model(self, train_dataset, val_dataset, num_epochs,
                    learning_rate, batch_size=32):
        print('Loading training data', end='\r')
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        print('Loaded Training data')
        print('Loading validation data', end='\r')
        val_loader = DataLoader(val_dataset, batch_size=batch_size)
        print('Loaded validation data  ')

        # loss_function = nn.NLLLoss()
        loss_function = nn.NLLLoss().to(device)  # Move loss function to GPU
        # Add weight decay (L2 regularization) to your optimizer to penalize large weights and prevent the model from overfitting.
        optimizer = optim.Adam(self.parameters(), lr=learning_rate, weight_decay=1e-5)
        
        # Move model to GPU
        self.to(device)
        print(f'model {device=}.')
        
        for epoch in tqdm(range(num_epochs), desc=f'Processing'):
            self.train() # method inherited from nn.Module that sets the model to training mode.
            total_train_loss = 0
            
            # The loops over train_loader and val_loader are necessary to handle batch-wise processing, so they cannot be fully vectorized.
            
            for context_embeddings, target_indices in train_loader:
                context_embeddings, target_indices = context_embeddings.to(device), target_indices.to(device)  # Move data to GPU
                self.zero_grad()
                log_probs = self(context_embeddings)  #  equivalent to calling self.forward(context_embeddings).
                loss  = loss_function(log_probs, target_indices)
                loss.backward()
                optimizer.step()
                total_train_loss += loss.item()
            
            self.eval()
            total_val_loss = 0
            with torch.no_grad():
                for context_embeddings, target_indices in val_loader:
                    context_embeddings, target_indices = context_embeddings.to(device), target_indices.to(device)  # Move data to GPU
                    log_probs = self(context_embeddings)
                    loss = loss_function(log_probs, target_indices)
                    total_val_loss += loss.item()
                
            print(f"Epoch {epoch+1}, Train Loss: {total_train_loss/len(train_loader):.4f}, Val Loss: {total_val_loss/len(val_loader):.4f}")
    
    def predict(self, context_embedding):
        self.eval()
        with torch.no_grad():
            context_embedding = context_embedding.to(device)  # Move data to GPU
            log_probs = self(context_embedding.unsqueeze(0))
        return torch.exp(log_probs)
    
    def perplexity(self, test_dataset):
        self.eval()
        test_loader = DataLoader(test_dataset, batch_size=1)
        total_loss = 0
        # loss_function = nn.NLLLoss()
        loss_function = nn.NLLLoss().to(device)  # Move loss function to GPU
        
        with torch.no_grad():
            for context_embeddings, target_indices in test_loader:
                context_embeddings, target_indices = context_embeddings.to(device), target_indices.to(device)  # Move data to GPU
                log_probs = self(context_embeddings)
                total_loss += loss_function(log_probs, target_indices).item()
        
        return np.exp(total_loss / len(test_dataset))
    
    def save_model(self, path):
        torch.save(self.state_dict(), path)
    
    def load_model(self, path):
        self.load_state_dict(torch.load(path))
        self.eval()
        
if __name__ == '__main__':
    pass
