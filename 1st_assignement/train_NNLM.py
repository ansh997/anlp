import os
import torch
from torch.utils.data import random_split
from collections import Counter
from sklearn.model_selection import train_test_split
import numpy as np
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from neural_language_model import TextDataset, create_dataloaders, ft_embedding, NLM  # some issue with this

scratch_location = '/scratch/hmnshpl/anlp_data'
filename = 'Auguste_Maquet.txt'
filepath = os.path.join(scratch_location, filename)
glove_file_path = '/scratch/hmnshpl/anlp_data/glove.6B.300d.txt' 
embedding_dim = 300

# Usage example:
# train_dataset, val_dataset, test_dataset = ...  # Your data loading code here
# vocab_size = ...  # Size of your vocabulary
# embedding_dim = ...  # Dimension of your embeddings

# results = NNLM.hyperparameter_tuning(train_dataset, val_dataset, test_dataset, vocab_size, embedding_dim)
# NNLM.plot_results(results)
# best_params = NNLM.find_best_hyperparameters(results)
# print("Best Hyperparameters:", best_params)

if __name__=='__main__':
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

    # Model parameters
    vocab_size = len(full_dataset.vocab)
    embedding_dim = full_dataset.embedding_dim
    hidden_dims = [300, 200]

    print(vocab_size, embedding_dim)

    # Initialize model
    print('Initializing model', end='\r')
    model = NLM.NNLM(vocab_size, embedding_dim, hidden_dims)
    print('Model Initialized    ')


    # Train model
    print('Training model', end='\r')
    model.train_model(train_dataset, val_dataset, num_epochs=10, learning_rate=0.001)
    print('model training done')


    # Evaluate
    print('Evaluating model', end=' ')
    test_perplexity = model.perplexity(test_dataset)
    print(f"Test Perplexity: {test_perplexity}")

    # Make a prediction
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=True)
    sample_context, _ = next(iter(test_loader))
    prediction = model.predict(sample_context)
    predicted_word_idx = prediction.argmax().item()
    predicted_word = full_dataset.idx_to_word[predicted_word_idx]
    print(f"Predicted word: {predicted_word}")
