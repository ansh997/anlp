import wandb
import torch.optim as optim
import warnings
import torch
import torch.nn as nn
import torch.optim as optim
from encoder import Encoder
from decoder import Decoder
from Old_dataset import get_dataloaders
import warnings
from train import train_epoch, evaluate, init_weights, Transformer

sweep_config = {
    'method': 'bayes',  # You can also use 'grid' or 'random'
    'metric': {'name': 'valid_loss', 'goal': 'minimize'},
    'parameters': {
        'num_layers': {'values': [4, 6, 8]},
        'num_heads': {'values': [4, 8, 16]},
        'model_dim': {'values': [256, 512, 1024]},
        'd_ff': {'values': [1024, 2048, 4096]},
        'dropout': {'values': [0.1, 0.3, 0.5]},
        'learning_rate': {'values': [0.00001, 0.0001, 0.001]},
        'batch_size': {'values': [16, 32, 64]},
        'scratch_location': {'value': "/scratch/hmnshpl/anlp_data/ted-talks-corpus"},
        'src_lang': {'value': 'en'},
        'tgt_lang': {'value': 'fr'},
        # 'tokenizer_type': {'value': ['bert', 'simple']},
    }
}



def train_model(epochs=10, max_len=100, tokenizer_type='simple'):
    wandb.init(),
    # Initialize wandb run
    config = wandb.config

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')
    
    # tokenizer_type = config.tokenizer_type

    # Get data loaders
    print('Loading data...', end='\r')
    train_loader, dev_loader, _ = get_dataloaders(
        config.scratch_location, config.src_lang, config.tgt_lang, 
        config.batch_size, max_len, tokenizer=tokenizer_type
    )
    print('Loaded data successfully.')

    # Instantiate encoder and decoder
    src_vocab_size = train_loader.dataset.src_vocab.__len__()
    tgt_vocab_size = train_loader.dataset.tgt_vocab.__len__()

    encoder = Encoder(
        src_vocab_size, config.model_dim, num_heads=config.num_heads, 
        num_layers=config.num_layers, d_ff=config.d_ff, max_seq_len=max_len, 
        dropout=config.dropout
    )
    decoder = Decoder(
        tgt_vocab_size, config.model_dim, num_heads=config.num_heads, 
        num_layers=config.num_layers, d_ff=config.d_ff, max_seq_len=max_len, 
        dropout=config.dropout
    )

    # Full transformer model
    model = Transformer(
        encoder, decoder, config.model_dim, src_vocab_size, tgt_vocab_size, max_len
    ).to(device)
    model.apply(init_weights)

    # Set up loss, optimizer, and scheduler
    criterion = nn.CrossEntropyLoss(
        ignore_index=train_loader.dataset.tgt_vocab.pad_idx, label_smoothing=0.1
    ).to(device)
    optimizer = optim.Adam(model.parameters(), lr=config.learning_rate, weight_decay=1e-6)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=2, gamma=0.1)

    best_valid_loss = float('inf')

    for epoch in range(epochs):
        # Train for one epoch
        train_loss = train_epoch(model, train_loader, criterion, optimizer, device, scheduler)
        print(f'Epoch {epoch+1}/{epochs} Training Loss: {train_loss:.4f}')

        # Validate the model
        tgt_vocab = train_loader.dataset.tgt_vocab
        valid_loss, bleu, rouge = evaluate(model, dev_loader, criterion, device, tgt_vocab)
        print(f'Validation Loss: {valid_loss:.4f}, BLEU: {bleu:.4f}, ROUGE-L: {rouge["rougeL"].fmeasure:.4f}')

        # Log the results in wandb
        wandb.log({
            "train_loss": train_loss,
            "valid_loss": valid_loss,
            "BLEU": bleu,
            "ROUGE-L": rouge["rougeL"].fmeasure,
            "epoch": epoch + 1
        })

        # Save the model if it has the best validation loss
        if valid_loss < best_valid_loss:
            best_valid_loss = valid_loss
            torch.save(model.state_dict(), 'transformer.pt')

    print('Training complete!')



if __name__ == '__main__':
    warnings.filterwarnings("ignore")
    
    sweep_id = wandb.sweep(sweep_config, project='ANLP_2nd_assignment')
    
    wandb.agent(sweep_id, function=train_model, count=10)
    
    
    

