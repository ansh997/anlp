import os
import torch
import wandb
from torch.utils.data import random_split
from neural_language_model import TextDataset, ft_embedding, NLM  # Ensure these are available

from skopt import BayesSearchCV
from skopt.space import Real, Categorical

# Set scratch location and other paths
scratch_location = '/scratch/hmnshpl/anlp_data'
filename = 'Auguste_Maquet.txt'
filepath = os.path.join(scratch_location, filename)
glove_file_path = '/scratch/hmnshpl/anlp_data/glove.6B.300d.txt'
embedding_dim = 300

# Initialize Weights and Biases (wandb)
wandb.init(project="NNLM-Bayesian-Optimization")

if __name__ == '__main__':
    print('Started')
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print('Available device: ', device)

    # Load embedding model
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

    def train_evaluate(hidden_dims, dropout_rate, learning_rate):
        """Train and evaluate model for given hyperparameters."""
        # Initialize model
        print('Initializing model', end='\r')
        model = NLM.NNLM(vocab_size, embedding_dim, hidden_dims, dropout_rate=dropout_rate).to(device)
        print('Model Initialized')

        # Train model
        print('Training model', end='\r')
        model.train_model(train_dataset, val_dataset, num_epochs=10, learning_rate=learning_rate)
        print('Model training done')

        # Evaluate model
        print('Evaluating model', end='\r')
        test_perplexity = model.perplexity(test_dataset)
        print(f"Test Perplexity: {test_perplexity}")

        # Log results to wandb
        wandb.log({
            "test_perplexity": test_perplexity,
            "hidden_dims": hidden_dims,
            "dropout_rate": dropout_rate,
            "learning_rate": learning_rate
        })

        return test_perplexity

    # Define search space for Bayesian Optimization
    search_space = {
        'hidden_dims': Categorical([[100, 200], [300, 200], [500, 300]]),
        'dropout_rate': Real(0.0, 0.5),
        'learning_rate': Real(1e-4, 1e-2, prior='log-uniform')
    }

    # Bayesian Optimization
    optimizer = BayesSearchCV(
        estimator=NLM.NNLM(vocab_size, embedding_dim, [300, 200]),  # Example placeholder model
        search_spaces=search_space,
        n_iter=25,  # Number of iterations
        scoring='neg_mean_squared_error',  # Placeholder, adjust to perplexity
        n_jobs=-1,
        cv=3
    )

    def objective(params):
        return train_evaluate(params['hidden_dims'], params['dropout_rate'], params['learning_rate'])

    # Perform Bayesian optimization
    results = optimizer.fit(train_dataset, test_dataset)
    best_params = optimizer.best_params_
    
    # Log the best results to wandb
    wandb.config.update(best_params)
    wandb.finish()
    
    print(f"Best Parameters: {best_params}")