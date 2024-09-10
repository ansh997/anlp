import os
import torch
from torch.utils.data import random_split
from collections import Counter
from sklearn.model_selection import train_test_split
from tqdm import tqdm
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
    
    sentence_perplexities = []
    output_file = f'{scratch_location}/2023701003_NNLM_perplexity_scores.txt'

    with open(output_file, 'w') as f:
        total_perplexity = 0.0
        for i, (sample_context, target) in tqdm(enumerate(test_loader), desc='Running'):
            
            word_indices = sample_context.argmax(dim=-1)
            
            # Get the sentence from context indices
            sentence = ' '.join([full_dataset.idx_to_word[idx.item()] for idx in sample_context.squeeze(0)])

            # Predict the word distribution and compute perplexity
            prediction = model.predict(sample_context)
            perplexity = model.compute_perplexity(prediction, target)

            # Log sentence perplexity
            f.write(f"{sentence}\t{perplexity.item()}\n")

            sentence_perplexities.append(perplexity.item())
            total_perplexity += perplexity.item()

        # Calculate and log average perplexity
        average_perplexity = total_perplexity / len(test_loader)
        f.write(f"\nAverage perplexity: {average_perplexity}\n")

    print(f"Perplexity scores written to {output_file}")
    
    
