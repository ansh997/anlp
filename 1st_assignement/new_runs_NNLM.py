import matplotlib.pyplot as plt
import numpy as np
import torch
from itertools import product
from torch import nn, optim
from torch.utils.data import DataLoader
import seaborn as sns

import os
from neural_language_model.NLM import NNLM
from neural_language_model import TextDataset, ft_embedding 
from torch.utils.data import random_split


scratch_location = '/scratch/hmnshpl/anlp_data'
filename = 'Auguste_Maquet.txt'
filepath = os.path.join(scratch_location, filename)
glove_file_path = '/scratch/hmnshpl/anlp_data/glove.6B.300d.txt' 
embedding_dim = 300

def hyperparameter_tuning_and_plot(train_dataset, val_dataset, test_dataset, vocab_size, embedding_dim):
    # Define the range of hyperparameters to tune
    hidden_dims = [[100], [200], [300], [100, 200], [200, 300], [300, 400]]
    num_hidden_layers = [1, 2]
    dropout_rates = [0.0, 0.2, 0.5]
    learning_rates = [0.001, 0.01]
    optimizers = [optim.SGD, optim.Adam]

    results = []
    
    # Iterate over all combinations of hyperparameters
    for hidden_dim, layers, dropout, lr, opt in product(hidden_dims, num_hidden_layers, dropout_rates, learning_rates, optimizers):
        print(f"Training with: hidden_dims={hidden_dim}, layers={layers}, dropout={dropout}, lr={lr}, optimizer={opt.__name__}")
        
        model = NNLM(vocab_size, embedding_dim, hidden_dim, num_hidden_layers=layers, dropout_rate=dropout)
        train_perplexities, val_perplexities = model.train_model(train_dataset, val_dataset, num_epochs=10, learning_rate=lr, optimizer_class=opt)
        test_perplexity = model.perplexity(test_dataset)
        
        results.append({
            'hidden_dims': hidden_dim,
            'num_layers': layers,
            'dropout_rate': dropout,
            'learning_rate': lr,
            'optimizer': opt.__name__,
            'train_perplexities': train_perplexities,
            'val_perplexities': val_perplexities,
            'test_perplexity': test_perplexity
        })

    # Plotting results
    plot_hyperparameter_results(results)
    
    # Report best hyperparameters
    best_hyperparameters = find_best_hyperparameters(results)
    print("Optimal Hyperparameters:", best_hyperparameters)
    return best_hyperparameters

def plot_hyperparameter_results(results):
    plt.figure(figsize=(15, 10))

    # Prepare seaborn for better aesthetics
    sns.set_theme(style="whitegrid")
    
    # Plot for each hyperparameter vs test perplexity
    for param in ['hidden_dims', 'num_layers', 'dropout_rate', 'learning_rate', 'optimizer']:
        plt.figure(figsize=(10, 6))
        
        if param in ['hidden_dims', 'num_layers', 'dropout_rate', 'learning_rate']:
            values = sorted(set(tuple(r[param]) if isinstance(r[param], list) else r[param] for r in results))
            test_perplexities = [
                                    min(r['test_perplexity'] for r in results if (tuple(r[param]) == val if isinstance(r[param], list) else r[param] == val))
                                    for val in values
                                ]

            plt.plot([str(val) for val in values], test_perplexities, marker='o')
        else:  # For optimizers, use a bar plot
            values = sorted(set(r[param] for r in results))
            test_perplexities = [min(r['test_perplexity'] for r in results if r[param] == val) for val in values]
            plt.bar(values, test_perplexities)

        plt.title(f'{param.replace("_", " ").title()} vs Test Perplexity')
        plt.xlabel(param.replace("_", " ").title())
        plt.ylabel('Best Test Perplexity')
        plt.xticks(rotation=45)
        plt.tight_layout()
        # plt.show()
        save_dir="hyperparameter_plots"
        # Save the plot
        plot_filename = os.path.join(save_dir, f'{param}_vs_test_perplexity.png')
        plt.savefig(plot_filename)
        plt.close()

    print(f"Plots saved in {save_dir}")

# Function to find and return the best hyperparameters
def find_best_hyperparameters(results):
    best_result = min(results, key=lambda r: r['test_perplexity'])
    return {k: v for k, v in best_result.items() if k != 'train_perplexities' and k != 'val_perplexities'}

if __name__ == '__main__':
    print('started')
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print('Available device: ', device)

    print('Loading embedding model', end='\r')
    embedding_model = ft_embedding.load_glove_embeddings(glove_file_path, embedding_dim)
    print('Loaded embedding model     ')

    # Create dataset
    full_dataset = TextDataset(filepath, embedding_model)

    # Split dataset
    train_size = int(0.7 * len(full_dataset))
    val_size = int(0.15 * len(full_dataset))
    test_size = len(full_dataset) - train_size - val_size
    train_dataset, val_dataset, test_dataset = random_split(full_dataset, [train_size, val_size, test_size])
    
    vocab_size = len(full_dataset.vocab)
    embedding_dim = full_dataset.embedding_dim
    hidden_dims = [300, 200]
    
    optimal_hyperparameters = hyperparameter_tuning_and_plot(train_dataset, val_dataset, test_dataset, vocab_size, embedding_dim)
