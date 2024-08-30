import torch
import torch.nn as nn
import torch.optim as optim

class LanguageModel(nn.Module):
    def __init__(self, embedding_dim, hidden_dim, vocab_size, num_hidden_layers):
        super(LanguageModel, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_hidden_layers = num_hidden_layers
        
        # Define the embedding layer
        self.embedding = nn.Linear(embedding_dim * 5, hidden_dim)
        
        # Create a list of hidden layers
        self.hidden_layers = nn.ModuleList()
        for _ in range(num_hidden_layers):
            self.hidden_layers.append(nn.Linear(hidden_dim, hidden_dim))
        
        # Define the output layer
        self.fc1 = nn.Linear(hidden_dim, vocab_size)
        
        # Define the softmax layer
        self.softmax = nn.Softmax(dim=1)
    
    def forward(self, x):
        # Flatten the input
        x = x.view(-1, 5 * self.hidden_dim)
        
        # Pass through the embedding layer
        x = torch.relu(self.embedding(x))
        
        # Pass through each hidden layer
        for layer in self.hidden_layers:
            x = torch.relu(layer(x))
        
        # Pass through the output layer
        x = self.fc1(x)
        
        # Apply softmax to get probabilities
        x = self.softmax(x)
        return x

# Example usage
embedding_dim = 300
hidden_dim = 300
vocab_size = 10  # len(embedding_model)  # The size of your vocabulary
num_hidden_layers = 3  # Number of hidden layers

model = LanguageModel(embedding_dim, hidden_dim, vocab_size, num_hidden_layers)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training loop (simplified example)
for epoch in range(10):  # Number of epochs
    for context, target in dataset:
        context = context.view(1, -1)  # Ensure context is in the correct shape for Linear layer
        optimizer.zero_grad()
        output = model(context)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
