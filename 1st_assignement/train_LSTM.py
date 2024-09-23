import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
import os
from torch.utils.data import random_split



from neural_language_model import ft_embedding
from neural_language_model.LSTMModel import LSTMLanguageModel, train_and_evaluate
from neural_language_model.preprocess_data import TextDataset

scratch_location = '/scratch/hmnshpl/anlp_data'
filename = 'Auguste_Maquet.txt'
filepath = os.path.join(scratch_location, filename)
glove_file_path = '/scratch/hmnshpl/anlp_data/glove.6B.300d.txt' 


def evaluate_and_save_perplexity(model, test_loader, full_dataset, loss_function,
                                device, output_file=f'{scratch_location}/2023701003_LSTM_perplexity_scores.txt'):
    model.eval()  # Set the model to evaluation mode
    total_perplexity = 0.0
    sentence_count = 0
    with open(output_file, 'w') as f:
        for sample_context, target in test_loader:
                        
            # Move data to the appropriate device (CPU/GPU)
            sample_context, target = sample_context.to(device), target.to(device)
            
            # Get the indices from the sample_context tensor
            indices = sample_context.squeeze(0).argmax(dim=-1)

            # Convert indices to words
            sentence = ' '.join([full_dataset.idx_to_word[idx.item()] for idx in indices])
            
            # Get the sentence from context indices
            # sentence = ' '.join([full_dataset.idx_to_word[idx.item()] for idx in sample_context.squeeze(0).argmax(dim=-1)])
            
            # Forward pass through the model to get predictions
            hidden = model.init_hidden(1, device)
            prediction, hidden = model(sample_context, hidden)
            # prediction = model.predict(sample_context, device) # model(sample_context)
            
            # Compute the loss and perplexity for the current sentence
            loss = loss_function(prediction, target)
            perplexity = torch.exp(loss).item()
            
            # Write sentence and perplexity to the file
            f.write(f"{sentence}\t{perplexity:.4f}\n")
            print(f"{sentence}\t{perplexity:.4f}\n")
            
            total_perplexity += perplexity
            sentence_count += 1
        
        # Calculate average perplexity
        average_perplexity = total_perplexity / sentence_count
        f.write(f"\nAverage perplexity: {average_perplexity:.4f}\n")
    
    print(f"Perplexity scores written to {output_file}")
    return average_perplexity


def main(args):
    embedding_dim = 300
    
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
    # hidden_dims = [300, 200]

    # print(vocab_size, embedding_dim)
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)
    
    # Initialize model
    model = LSTMLanguageModel(embedding_dim, args.hidden_dim, vocab_size, args.num_layers)
    
    # Optimizer and loss function
    # Add weight decay (L2 regularization) to your optimizer to penalize large weights and prevent the model from overfitting.
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=1e-5)
    loss_function = nn.CrossEntropyLoss()
    
    # Train and evaluate the model
    train_and_evaluate(model, train_loader, val_loader, optimizer, loss_function, args.num_epochs, args.device)
    
    # Calculate and report perplexity on the test set
    test_perplexity = model.perplexity(test_loader, loss_function, args.device)
    print(f'Test Perplexity: {test_perplexity:.4f}')

    # Make a prediction
    sample_context, _ = next(iter(test_loader))
    predicted_word_idx = model.predict(sample_context, args.device)
    predicted_word = full_dataset.idx_to_word[predicted_word_idx]
    print(f'Predicted word: {predicted_word}')
    
    output_file = f'{scratch_location}/2023701003_LSTM_perplexity_scores.txt'
    average_perplexity = evaluate_and_save_perplexity(model, test_loader, full_dataset, loss_function, args.device, output_file)
    print(f'Test Average Perplexity: {average_perplexity:.4f}')

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train an LSTM-based Language Model')
    
    # parser.add_argument('--embedding_model', type=dict, required=True, help='Pre-trained word embeddings model')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size for training and evaluation')
    parser.add_argument('--hidden_dim', type=int, default=300, help='Hidden dimension of LSTM')
    parser.add_argument('--num_layers', type=int, default=1, help='Number of LSTM layers')
    parser.add_argument('--learning_rate', type=float, default=0.001, help='Learning rate for the optimizer')
    parser.add_argument('--num_epochs', type=int, default=10, help='Number of epochs for training')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu', help='Device to run the model on (cuda or cpu)')
    
    args = parser.parse_args()
    
    main(args)