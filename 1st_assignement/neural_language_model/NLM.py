import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
import matplotlib.pyplot as plt
from itertools import product

class NNLM(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dims, num_hidden_layers=2, context_size=5, dropout_rate=0.5):
        super(NNLM, self).__init__()

        assert isinstance(context_size, int), f"Expected context_size to be int but got {type(context_size)}"
        assert isinstance(embedding_dim, int), f"Expected embedding_dim to be int but got {type(embedding_dim)}"
        assert isinstance(hidden_dims, list), f"Expected hidden_dim to be list but got {type(hidden_dims)}"
        assert len(hidden_dims) == num_hidden_layers, "num_hidden_layers should be equal to length of hidden_dims"

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')  # Device configuration

        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dims
        self.context_size = context_size
        self.num_hidden_layers = num_hidden_layers

        self.input_dim = context_size * embedding_dim

        print(f"context_size: {context_size}, embedding_dim: {embedding_dim}, hidden_dim: {hidden_dims}")

        # Create variable number of hidden layers
        layers = []
        in_features = self.input_dim
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(in_features, hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout_rate))
            in_features = hidden_dim

        layers.append(nn.Linear(hidden_dims[-1], vocab_size))

        self.model = nn.Sequential(*layers)
        self.softmax = nn.LogSoftmax(dim=1)

        # Move model to the appropriate device
        self.to(self.device)

    def forward(self, inputs):
        # Ensure inputs are on the same device as the model
        inputs = inputs.to(self.device)
        flattened_embeds = inputs.view(inputs.shape[0], -1)
        output = self.model(flattened_embeds)
        return self.softmax(output)

    def train_model(self, train_dataset, val_dataset, num_epochs, learning_rate, batch_size=32, optimizer_class=optim.Adam):
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size)

        loss_function = nn.NLLLoss()
        optimizer = optimizer_class(self.parameters(), lr=learning_rate, weight_decay=1e-3)

        train_perplexities = []
        val_perplexities = []

        for epoch in range(num_epochs):
            self.train()
            total_train_loss = 0
            for context_embeddings, target_indices in train_loader:
                context_embeddings, target_indices = context_embeddings.to(self.device), target_indices.to(self.device)
                
                self.zero_grad()
                log_probs = self(context_embeddings)
                loss = loss_function(log_probs, target_indices)
                loss.backward()
                optimizer.step()
                total_train_loss += loss.item()

            train_perplexity = np.exp(total_train_loss / len(train_loader))
            train_perplexities.append(train_perplexity)

            # Validation
            self.eval()
            total_val_loss = 0
            with torch.no_grad():
                for context_embeddings, target_indices in val_loader:
                    context_embeddings, target_indices = context_embeddings.to(self.device), target_indices.to(self.device)
                    
                    log_probs = self(context_embeddings)
                    loss = loss_function(log_probs, target_indices)
                    total_val_loss += loss.item()

            val_perplexity = np.exp(total_val_loss / len(val_loader))
            val_perplexities.append(val_perplexity)

            print(f"Epoch {epoch+1}, Train Perplexity: {train_perplexity:.4f}, Val Perplexity: {val_perplexity:.4f}")

        return train_perplexities, val_perplexities

    def predict(self, context_embedding):
        self.eval()
        with torch.no_grad():
            # Ensure context_embedding is on the right device
            context_embedding = context_embedding.to(self.device)
            log_probs = self(context_embedding.unsqueeze(0))
        return torch.exp(log_probs)

    def perplexity(self, test_dataset):
        self.eval()
        test_loader = DataLoader(test_dataset, batch_size=1)
        total_loss = 0
        loss_function = nn.NLLLoss()

        with torch.no_grad():
            for context_embeddings, target_indices in test_loader:
                context_embeddings, target_indices = context_embeddings.to(self.device), target_indices.to(self.device)
                log_probs = self(context_embeddings)
                total_loss += loss_function(log_probs, target_indices).item()

        return np.exp(total_loss / len(test_dataset))

    @staticmethod
    def hyperparameter_tuning(train_dataset, val_dataset, test_dataset, vocab_size, embedding_dim):
        hidden_dims = [100, 200, 300]
        num_hidden_layers = [1, 2, 3]
        dropout_rates = [0.0, 0.2, 0.5]
        learning_rates = [0.001, 0.01, 0.1]
        optimizers = [optim.SGD, optim.Adam]

        results = []

        for hidden_dim, layers, dropout, lr, opt in product(hidden_dims, num_hidden_layers, dropout_rates, learning_rates, optimizers):
            print(f"Training with: hidden_dim={hidden_dim}, layers={layers}, dropout={dropout}, lr={lr}, optimizer={opt.__name__}")

            model = NNLM(vocab_size, embedding_dim, hidden_dim, num_hidden_layers=layers, dropout_rate=dropout)
            train_perp, val_perp = model.train_model(train_dataset, val_dataset, num_epochs=10, learning_rate=lr, optimizer_class=opt)
            test_perp = model.perplexity(test_dataset)

            results.append({
                'hidden_dim': hidden_dim,
                'num_layers': layers,
                'dropout_rate': dropout,
                'learning_rate': lr,
                'optimizer': opt.__name__,
                'train_perplexities': train_perp,
                'val_perplexities': val_perp,
                'test_perplexity': test_perp
            })

        return results

    @staticmethod
    def plot_results(results):
        fig, axs = plt.subplots(2, 3, figsize=(20, 15))

        params = ['hidden_dim', 'num_layers', 'dropout_rate', 'learning_rate', 'optimizer']
        for i, param in enumerate(params):
            row = i // 3
            col = i % 3

            if param in ['hidden_dim', 'num_layers', 'dropout_rate', 'learning_rate']:
                x = sorted(set(r[param] for r in results))
                y = [min(r['test_perplexity'] for r in results if r[param] == val) for val in x]
                axs[row, col].plot(x, y, 'o-')
                if param == 'learning_rate':
                    axs[row, col].set_xscale('log')
            else:  # optimizer
                x = sorted(set(r[param] for r in results))
                y = [min(r['test_perplexity'] for r in results if r[param] == val) for val in x]
                axs[row, col].bar(x, y)

            axs[row, col].set_xlabel(param.replace('_', ' ').title())
            axs[row, col].set_ylabel('Best Test Perplexity')
            axs[row, col].set_title(f'{param.replace("_", " ").title()} vs Test Perplexity')

        plt.tight_layout()
        plt.savefig('hyperparameter_tuning_results.png')
        plt.close()

    @staticmethod
    def find_best_hyperparameters(results):
        best_result = min(results, key=lambda r: r['test_perplexity'])
        return {k: v for k, v in best_result.items() if k != 'train_perplexities' and k != 'val_perplexities'}