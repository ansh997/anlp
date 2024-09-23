import os
import torch
import wandb
from torch.utils.data import random_split
from tqdm import tqdm
from torch.utils.data import DataLoader
from neural_language_model import TextDataset, ft_embedding, NLM  # Make sure to import your model

# Set file paths
scratch_location = '/scratch/hmnshpl/anlp_data'
filename = 'Auguste_Maquet.txt'
filepath = os.path.join(scratch_location, filename)
glove_file_path = '/scratch/hmnshpl/anlp_data/glove.6B.300d.txt'
embedding_dim = 300

# Sweep configuration
sweep_config = {
    'method': 'bayes',
    'metric': {
        'name': 'test_perplexity',
        'goal': 'minimize'
    },
    'parameters': {
        'hidden_dims': {
            'values': [[100, 200], [300, 200], [500, 300]]
        },
        'dropout_rate': {
            'min': 0.0,
            'max': 0.5
        },
        'learning_rate': {
            'min': 1e-4,
            'max': 1e-2
        }
    }
}

# Define the training function
def train():
    # Initialize a new wandb run
    wandb.init()

    # Get hyperparameters from wandb
    config = wandb.config

    # Load the embedding model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    embedding_model = ft_embedding.load_glove_embeddings(glove_file_path, embedding_dim)

    # Create the dataset
    full_dataset = TextDataset(filepath, embedding_model)

    # Split dataset
    train_size = int(0.7 * len(full_dataset))
    val_size = int(0.15 * len(full_dataset))
    test_size = len(full_dataset) - train_size - val_size
    train_dataset, val_dataset, test_dataset = random_split(full_dataset, [train_size, val_size, test_size])

    # Define model parameters
    vocab_size = len(full_dataset.vocab)
    embedding_dim = full_dataset.embedding_dim

    # Initialize model with current hyperparameters
    model = NLM.NNLM(vocab_size, embedding_dim, config.hidden_dims, dropout_rate=config.dropout_rate).to(device)

    # Train the model
    model.train_model(train_dataset, val_dataset, num_epochs=10, learning_rate=config.learning_rate)

    # Evaluate the model on the test dataset and calculate perplexity
    test_perplexity = model.perplexity(test_dataset)

    # Log the test perplexity to wandb
    wandb.log({'test_perplexity': test_perplexity})

# Create the sweep and start running
if __name__ == '__main__':
    # Initialize sweep
    sweep_id = wandb.sweep(sweep_config, project="NNLM-Bayesian-Optimization")

    # Start sweep
    wandb.agent(sweep_id, train, count=20)  # Specify how many sweep runs to execute
