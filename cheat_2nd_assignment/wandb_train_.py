from model import build_transformer
from dataset import BLDataset, causal_mask
from config import get_weights_file_path, scratch_path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
# from torch.optim.lr_scheduler import LambdaLR

import warnings
from tqdm import tqdm
import os
from pathlib import Path

# Huggingface datasets and tokenizers
# from datasets import load_dataset
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.trainers import WordLevelTrainer
from tokenizers.pre_tokenizers import Whitespace

import wandb

import torchmetrics


sweep_config = {
        'method': 'bayes',
        'metric': {'name': 'valid_loss', 'goal': 'minimize'},
        'parameters': {
            'num_layers': {'values': [2, 4]},
            'num_heads': {'values': [2, 4, 8]},
            'd_model': {'values': [256, 512]},
            'd_ff': {'values': [128]},
            'dropout': {'values': [0.1, 0.3, 0.5]},
            'lr': {'values': [0.00001, 0.0001, 0.001]},
            'batch_size': {'values': [8, 16]},
            'datasource': {'value': "/scratch/hmnshpl/anlp_data/ted-talks-corpus"},
            'lang_src': {'value': 'en'},
            'lang_tgt': {'value': 'fr'},
            "model_folder": {'value': f"{scratch_path}/weights"},
            "model_basename": {'value': "tmodel_"},
            "preload": {'value': None},
            "tokenizer_file": {'value': "/scratch/hmnshpl/anlp_data/tokenizer_{0}.json"},
            "experiment_name": {'value': f"{scratch_path}/runs/tmodel"},
            "seq_len": {'value': 350},
            "num_epochs": {'value': 10}
        }
    }

def greedy_decode(model, source, source_mask, tokenizer_tgt, max_len, device):
    sos_idx = tokenizer_tgt.token_to_id('[SOS]')
    eos_idx = tokenizer_tgt.token_to_id('[EOS]')

    # Precompute the encoder output and reuse it for every step
    encoder_output = model.encode(source, source_mask)
    # Initialize the decoder input with the sos token
    decoder_input = torch.empty(1, 1).fill_(sos_idx).type_as(source).to(device)
    while True:
        if decoder_input.size(1) == max_len:
            break

        # build mask for target
        decoder_mask = causal_mask(decoder_input.size(1)).type_as(source_mask).to(device)

        # calculate output
        out = model.decode(encoder_output, source_mask, decoder_input, decoder_mask)

        # get next token
        prob = model.project(out[:, -1])
        _, next_word = torch.max(prob, dim=1)
        decoder_input = torch.cat(
            [decoder_input, torch.empty(1, 1).type_as(source).fill_(next_word.item()).to(device)], dim=1
        )

        if next_word == eos_idx:
            break

    return decoder_input.squeeze(0)


def _run_validation(model, validation_ds, tokenizer_src, tokenizer_tgt, max_len, device, print_msg, global_step, num_examples=2):
    model.eval()
    count = 0

    source_texts = []
    expected = []
    predicted = []

    try:
        # get the console window width
        with os.popen('stty size', 'r') as console:
            _, console_width = console.read().split()
            console_width = int(console_width)
    except:
        # If we can't get the console width, use 80 as default
        console_width = 80

    with torch.no_grad():
        for batch in validation_ds:
            count += 1
            encoder_input = batch["encoder_input"].to(device) # (b, seq_len)
            encoder_mask = batch["encoder_mask"].to(device) # (b, 1, 1, seq_len)

            # check that the batch size is 1
            assert encoder_input.size(
                0) == 1, "Batch size must be 1 for validation"

            # greedy_decode(model, source, source_mask, tokenizer_tgt, max_len, device)
            model_out = greedy_decode(model, encoder_input, encoder_mask, tokenizer_tgt, max_len, device)

            source_text = batch["src_text"][0]
            target_text = batch["tgt_text"][0]
            model_out_text = tokenizer_tgt.decode(model_out.detach().cpu().numpy())

            source_texts.append(source_text)
            expected.append(target_text)
            predicted.append(model_out_text)
            
            # Print the source, target and model output
            print_msg('-'*console_width)
            print_msg(f"{f'SOURCE: ':>12}{source_text}")
            print_msg(f"{f'TARGET: ':>12}{target_text}")
            print_msg(f"{f'PREDICTED: ':>12}{model_out_text}")

            if count == num_examples:
                print_msg('-'*console_width)
                break
    
    
    # Evaluate the character error rate
    # Compute the char error rate 
    metric = torchmetrics.CharErrorRate()
    cer = metric(predicted, expected)
    wandb.log({'validation/cer': cer, 'global_step': global_step})

    # Compute the word error rate
    metric = torchmetrics.WordErrorRate()
    wer = metric(predicted, expected)
    wandb.log({'validation/wer': wer, 'global_step': global_step})

    # Compute the BLEU metric
    metric = torchmetrics.BLEUScore()
    bleu = metric(predicted, expected)
    wandb.log({'validation/BLEU': bleu, 'global_step': global_step})
    
    return cer, wer, bleu


def run_validation(model, validation_ds, tokenizer_src, tokenizer_tgt, max_len, device, print_msg, global_step, loss_fn, num_examples=2):
    model.eval()
    count = 0

    source_texts = []
    expected = []
    predicted = []

    total_valid_loss = 0.0  # To accumulate validation loss
    num_batches = 0

    try:
        # get the console window width
        with os.popen('stty size', 'r') as console:
            _, console_width = console.read().split()
            console_width = int(console_width)
    except:
        # If we can't get the console width, use 80 as default
        console_width = 80

    with torch.no_grad():
        for batch in validation_ds:
            count += 1
            encoder_input = batch["encoder_input"].to(device)  # (b, seq_len)
            encoder_mask = batch["encoder_mask"].to(device)  # (b, 1, 1, seq_len)
            label = batch['label'].to(device)  # Ground truth target

            # Check that the batch size is 1
            assert encoder_input.size(0) == 1, "Batch size must be 1 for validation"

            # Greedy decoding
            model_out = greedy_decode(model, encoder_input, encoder_mask, tokenizer_tgt, max_len, device)

            # Calculate validation loss
            proj_output = model.project(model_out)  # (B, seq_len, vocab_size)
            valid_loss = loss_fn(proj_output.view(-1, tokenizer_tgt.get_vocab_size()), label.view(-1))
            total_valid_loss += valid_loss.item()
            num_batches += 1

            # Convert output to text
            source_text = batch["src_text"][0]
            target_text = batch["tgt_text"][0]
            model_out_text = tokenizer_tgt.decode(model_out.detach().cpu().numpy())

            source_texts.append(source_text)
            expected.append(target_text)
            predicted.append(model_out_text)
            
            # Print the source, target, and model output
            print_msg('-' * console_width)
            print_msg(f"{f'SOURCE: ':>12}{source_text}")
            print_msg(f"{f'TARGET: ':>12}{target_text}")
            print_msg(f"{f'PREDICTED: ':>12}{model_out_text}")

            if count == num_examples:
                print_msg('-' * console_width)
                break
    
    # Calculate and log the average validation loss
    avg_valid_loss = total_valid_loss / num_batches
    wandb.log({'validation/loss': avg_valid_loss, 'global_step': global_step})

    # Evaluate the character error rate (CER)
    metric = torchmetrics.CharErrorRate()
    cer = metric(predicted, expected)
    wandb.log({'validation/cer': cer, 'global_step': global_step})

    # Evaluate the word error rate (WER)
    metric = torchmetrics.WordErrorRate()
    wer = metric(predicted, expected)
    wandb.log({'validation/wer': wer, 'global_step': global_step})

    # Evaluate the BLEU score
    metric = torchmetrics.BLEUScore()
    bleu = metric(predicted, expected)
    wandb.log({'validation/BLEU': bleu, 'global_step': global_step})
    
    return cer, wer, bleu, avg_valid_loss



def get_all_sentences(ds, lang):
    for item in ds:
        yield item['translation'][lang]


def read_file(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        return [line.strip() for line in f]
    

def get_or_build_tokenizer(config, sentences, lang):
    tokenizer_path = Path(config['tokenizer_file'].format(lang))
    if not Path.exists(tokenizer_path):
        tokenizer = Tokenizer(WordLevel(unk_token="[UNK]"))
        tokenizer.pre_tokenizer = Whitespace()
        trainer = WordLevelTrainer(special_tokens=["[UNK]", "[PAD]", "[SOS]", "[EOS]"], min_frequency=2)
        # tokenizer.train_from_iterator(get_all_sentences(ds, lang), trainer=trainer)
        tokenizer.train_from_iterator(sentences, trainer=trainer)
        tokenizer.save(str(tokenizer_path))
    else:
        tokenizer = Tokenizer.from_file(str(tokenizer_path))
    return tokenizer

def get_ds(config):
    # Load the datasets
    train_src = read_file(os.path.join(config['datasource'], f'train.{config["lang_src"]}'))
    train_tgt = read_file(os.path.join(config['datasource'], f'train.{config["lang_tgt"]}'))
    val_src = read_file(os.path.join(config['datasource'], f'dev.{config["lang_src"]}'))
    val_tgt = read_file(os.path.join(config['datasource'], f'dev.{config["lang_tgt"]}'))

    # Build tokenizers
    tokenizer_src = get_or_build_tokenizer(config, train_src, config['lang_src'])
    tokenizer_tgt = get_or_build_tokenizer(config, train_tgt, config['lang_tgt'])

    # Create datasets
    train_ds = BLDataset(train_src, train_tgt, tokenizer_src, tokenizer_tgt, config['lang_src'], config['lang_tgt'], config['seq_len'])
    val_ds = BLDataset(val_src, val_tgt, tokenizer_src, tokenizer_tgt, config['lang_src'], config['lang_tgt'], config['seq_len'])


    # Create data loaders
    train_dataloader = DataLoader(train_ds, batch_size=config['batch_size'], shuffle=True)
    val_dataloader = DataLoader(val_ds, batch_size=1, shuffle=False)

    return train_dataloader, val_dataloader, tokenizer_src, tokenizer_tgt

def get_model(config, vocab_src_len, vocab_tgt_len):
    model = build_transformer(vocab_src_len, vocab_tgt_len, config["seq_len"], config['seq_len'], d_model=config['d_model'])
    return model

def train_model():
    
    wandb.init(dir=scratch_path)
    
    config = wandb.config
    # print(config)
    # Define the device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    # Make sure the weights folder exists
    Path(config['model_folder']).mkdir(parents=True, exist_ok=True)

    train_dataloader, val_dataloader, tokenizer_src, tokenizer_tgt = get_ds(config)
    model = get_model(config, tokenizer_src.get_vocab_size(), tokenizer_tgt.get_vocab_size()).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=config['lr'], eps=1e-9)

    # If the user specified a model to preload before training, load it
    initial_epoch = 0
    global_step = 0
    if config['preload']:
        model_filename = get_weights_file_path(config, config['preload'])
        print(f'Preloading model {model_filename}')
        state = torch.load(model_filename)
        model.load_state_dict(state['model_state_dict'])
        initial_epoch = state['epoch'] + 1
        optimizer.load_state_dict(state['optimizer_state_dict'])
        global_step = state['global_step']
        del state

    loss_fn = nn.CrossEntropyLoss(ignore_index=tokenizer_src.token_to_id('[PAD]'), label_smoothing=0.1).to(device)


    for epoch in range(initial_epoch, config['num_epochs']):
        torch.cuda.empty_cache()
        model.train()
        batch_iterator = tqdm(train_dataloader, desc=f"Processing Epoch {epoch:02d}", leave=False, ncols=100)
        total_train_loss = 0
        num_batches = len(train_dataloader)

        for _, batch in enumerate(batch_iterator):

            encoder_input = batch['encoder_input'].to(device) # (b, seq_len)
            decoder_input = batch['decoder_input'].to(device) # (B, seq_len)
            encoder_mask = batch['encoder_mask'].to(device) # (B, 1, 1, seq_len)
            decoder_mask = batch['decoder_mask'].to(device) # (B, 1, seq_len, seq_len)

            # Run the tensors through the encoder, decoder and the projection layer
            encoder_output = model.encode(encoder_input, encoder_mask) # (B, seq_len, d_model)
            decoder_output = model.decode(encoder_output, encoder_mask, decoder_input, decoder_mask) # (B, seq_len, d_model)
            proj_output = model.project(decoder_output) # (B, seq_len, vocab_size)

            # Compare the output with the label
            label = batch['label'].to(device) # (B, seq_len)

            # Compute the loss using a simple cross entropy
            loss = loss_fn(proj_output.view(-1, tokenizer_tgt.get_vocab_size()), label.view(-1))
            total_train_loss += loss.item()
            batch_iterator.set_postfix({f"loss @ epoch {epoch}": f"{loss.item():6.3f}"})

            # # Log the loss
            # if global_step % 10 == 0:
            #     wandb.log({'train/loss': loss.item(), 'global_step': global_step})

            # Backpropagate the loss
            loss.backward()

            # Update the weights
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            

            global_step += 1
        
        avg_train_loss = total_train_loss / num_batches
        wandb.log({'train/avg_loss': avg_train_loss, 'epoch': epoch})
        print(f"Epoch {epoch}, Avg Train Loss: {avg_train_loss}")

        # Run validation at the end of every epoch
        cer, wer, bleu, valid_loss = run_validation(model, val_dataloader, tokenizer_src, tokenizer_tgt, config['seq_len'], device, lambda msg: batch_iterator.write(msg), global_step, loss_fn=loss_fn)
        wandb.log({'cer': cer, 'epoch': epoch})
        wandb.log({'wer': wer, 'epoch': epoch})
        wandb.log({'bleu': bleu, 'epoch': epoch})
        wandb.log({'valid_loss': valid_loss, 'epoch': epoch})

        # Save the model at the end of every epoch
        model_filename = get_weights_file_path(config, f"{epoch:02d}")
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'global_step': global_step
        }, model_filename)


if __name__ == '__main__':
    warnings.filterwarnings("ignore")
    # config = sweep_config
    
    # Initialize the sweep and run it
    sweep_id = wandb.sweep(sweep_config, project='Assignment2 - Transformer')
    wandb.agent(sweep_id, train_model, count=10)