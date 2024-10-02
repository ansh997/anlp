import wandb
import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from pathlib import Path

from config import get_weights_file_path, latest_weights_file_path, scratch_path
from train import get_ds, get_model, run_validation

def train_model():
    # Initialize a new wandb run with the current hyperparameters
    wandb.init(dir=f'{scratch_path}')

    config = wandb.config  # This will capture the config parameters from wandb sweep

    # Define the device
    device = "cuda" if torch.cuda.is_available() else "mps" if torch.has_mps or torch.backends.mps.is_available() else "cpu"
    print("Using device:", device)
    
    device = torch.device(device)

    # Make sure the weights folder exists
    Path(f"{config.datasource}_{config.model_folder}").mkdir(parents=True, exist_ok=True)

    # Get data and model
    train_dataloader, val_dataloader, tokenizer_src, tokenizer_tgt = get_ds(config)
    model = get_model(config, tokenizer_src.get_vocab_size(), tokenizer_tgt.get_vocab_size()).to(device)
    
    # Optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate, eps=1e-9)

    # Preload model if applicable
    initial_epoch = 0
    global_step = 0
    preload = config.preload
    model_filename = latest_weights_file_path(config) if preload == 'latest' else get_weights_file_path(config, preload) if preload else None
    if model_filename:
        print(f'Preloading model {model_filename}')
        state = torch.load(model_filename)
        model.load_state_dict(state['model_state_dict'])
        initial_epoch = state['epoch'] + 1
        optimizer.load_state_dict(state['optimizer_state_dict'])
        global_step = state['global_step']
    else:
        print('No model to preload, starting from scratch')

    # Loss function
    loss_fn = nn.CrossEntropyLoss(ignore_index=tokenizer_src.token_to_id('[PAD]'), label_smoothing=0.1).to(device)

    # Tensorboard writer
    writer = SummaryWriter(config['experiment_name'])

    accumulation_steps = config.get('accumulation_steps', 1)  # Set default to 1 if not provided

    # Training loop
    for epoch in range(initial_epoch, config.num_epochs):
        torch.cuda.empty_cache()
        model.train()
        batch_iterator = tqdm(train_dataloader, desc=f"Processing Epoch {epoch:02d}", leave=False, ncols=80)
        epoch_loss = 0

        optimizer.zero_grad(set_to_none=True)

        for i, batch in enumerate(batch_iterator):
            encoder_input = batch['encoder_input'].to(device)
            decoder_input = batch['decoder_input'].to(device)
            encoder_mask = batch['encoder_mask'].to(device)
            decoder_mask = batch['decoder_mask'].to(device)

            # Forward pass
            encoder_output = model.encode(encoder_input, encoder_mask)
            decoder_output = model.decode(encoder_output, encoder_mask, decoder_input, decoder_mask)
            proj_output = model.project(decoder_output)

            label = batch['label'].to(device)
            loss = loss_fn(proj_output.view(-1, tokenizer_tgt.get_vocab_size()), label.view(-1))
            epoch_loss += loss.item()

            # Backpropagate the loss, but don't update the weights yet
            loss.backward()

            # Gradient accumulation: update weights only after accumulating enough gradients
            if (i + 1) % accumulation_steps == 0:
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)

            # Log loss to wandb
            wandb.log({'train_loss': loss.item(), 'epoch': epoch})

        # Validation at the end of every epoch
        valid_loss = run_validation(model, val_dataloader, tokenizer_src, tokenizer_tgt, config.seq_len, device)
        wandb.log({'valid_loss': valid_loss, 'epoch': epoch})

        # Save model
        model_filename = get_weights_file_path(config, f"{epoch:02d}")
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'global_step': global_step
        }, model_filename)

    writer.close()

if __name__ == '__main__':
    sweep_config = {
        'method': 'bayes',
        'metric': {'name': 'valid_loss', 'goal': 'minimize'},
        'parameters': {
            'num_layers': {'values': [4, 6]},
            'num_heads': {'values': [4, 8]},
            'd_model': {'values': [512]},
            'd_ff': {'values': [512, 1024]},
            'dropout': {'values': [0.1, 0.3, 0.5]},
            'learning_rate': {'values': [0.00001, 0.0001, 0.001]},
            'batch_size': {'values': [16, 32]},
            'datasource': {'value': "/scratch/hmnshpl/anlp_data/ted-talks-corpus"},
            'lang_src': {'value': 'en'},
            'lang_tgt': {'value': 'fr'},
            "model_folder": {'value': f"{scratch_path}/weights"},
            "model_basename": {'value': "tmodel_"},
            "preload": {'value': "latest"},
            "tokenizer_file": {'value': "/scratch/hmnshpl/anlp_data/tokenizer_{0}.json"},
            "experiment_name": {'value': f"{scratch_path}/runs/tmodel"},
            "seq_len": {'value': 350},
            "num_epochs": {'value': 10}
        }
    }

    # Initialize the sweep and run it
    sweep_id = wandb.sweep(sweep_config, project='Assignment2 - Transformer')
    wandb.agent(sweep_id, train_model, count=10)