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
from train import init_weights, Transformer
from tqdm.auto import tqdm
import sacrebleu
from rouge_score import rouge_scorer

# sweep_config = {
#     'method': 'bayes',  # You can also use 'grid' or 'random'
#     'metric': {'name': 'valid_loss', 'goal': 'minimize'},
#     'parameters': {
#         'num_layers': {'values': [4, 6, 8]},
#         'num_heads': {'values': [4, 8, 16]},
#         'model_dim': {'values': [256, 512, 1024]},
#         'd_ff': {'values': [1024, 2048, 4096]},
#         'dropout': {'values': [0.1, 0.3, 0.5]},
#         'learning_rate': {'values': [0.00001, 0.0001, 0.001]},
#         'batch_size': {'values': [16, 32, 64]},
#         'scratch_location': {'value': "/scratch/hmnshpl/anlp_data/ted-talks-corpus"},
#         'src_lang': {'value': 'en'},
#         'tgt_lang': {'value': 'fr'},
#         # 'tokenizer_type': {'value': ['bert', 'simple']},
#     }
# }


sweep_config = {
    'method': 'bayes',  # You can also use 'grid' or 'random'
    'metric': {'name': 'valid_loss', 'goal': 'minimize'},
    'parameters': {
        'num_layers': {'values': [6]},
        'num_heads': {'values': [8]},
        'model_dim': {'values': [512]},
        'd_ff': {'values': [1024]},
        'dropout': {'values': [0.1]},
        'learning_rate': {'values': [0.00001]},
        'batch_size': {'values': [32]},
        'scratch_location': {'value': "/scratch/hmnshpl/anlp_data/ted-talks-corpus"},
        'src_lang': {'value': 'en'},
        'tgt_lang': {'value': 'fr'},
        # 'tokenizer_type': {'value': ['bert', 'simple']},
    }
}


def evaluate(model, dataloader, criterion, device, tgt_vocab, max_len):
    model.eval()
    epoch_loss = 0
    all_preds = []
    all_refs = []

    scorer_bleu = sacrebleu.metrics.BLEU()
    rouge_scorer_obj = rouge_scorer.RougeScorer(['rouge1', 'rougeL'], use_stemmer=True)

    start_symbol = tgt_vocab.unk_idx + 1  # Adjust this if needed

    with torch.no_grad():
        for src_batch, tgt_batch in tqdm(dataloader, desc='Evaluating', leave=False, ncols=80):
            src_batch, tgt_batch = src_batch.to(device), tgt_batch.to(device)

            # Shift target tokens for teacher forcing
            tgt_input = tgt_batch[:, :-1]
            tgt_output = tgt_batch[:, 1:]

            # Forward pass for batch
            output = model(src_batch, tgt_input)
            output_flat = output.contiguous().view(-1, output.size(-1))
            tgt_output_flat = tgt_output.contiguous().view(-1)

            # Calculate loss
            loss = criterion(output_flat, tgt_output_flat)
            epoch_loss += loss.item()

            # Batch prediction generation
            preds = generate_batch_predictions(model, src_batch, max_len, start_symbol, device)

            # Decode token indices to text in batch
            preds_text = tgt_vocab.batch_decode([pred.cpu().tolist() for pred in preds])
            refs_text = [[tgt_vocab.decode(ref[1:].cpu().tolist())] for ref in tgt_batch]  # Skip the start token

            all_preds.extend(preds_text)
            all_refs.extend(refs_text)

    # Compute BLEU and ROUGE scores in batch
    bleu_score = scorer_bleu.corpus_score(all_preds, all_refs).score

    rouge_scores = {'rouge1': [], 'rougeL': []}
    for pred, ref in zip(all_preds, all_refs):
        scores = rouge_scorer_obj.score(ref[0], pred)
        for key in rouge_scores:
            rouge_scores[key].append(scores[key].fmeasure)

    # Average ROUGE scores
    rouge_scores = {key: sum(scores) / len(scores) for key, scores in rouge_scores.items()}

    return epoch_loss / len(dataloader), bleu_score, rouge_scores


def generate_batch_predictions(model, src_batch, max_len, start_symbol, device):
    """
    Efficient batch prediction generation using greedy decoding.
    """
    batch_size = src_batch.size(0)
    preds = torch.full((batch_size, 1), start_symbol, dtype=torch.long, device=device)
    
    for _ in range(max_len):
        output = model(src_batch, preds)  # Forward pass
        next_tokens = output[:, -1, :].argmax(dim=-1, keepdim=True)  # Get next token predictions
        preds = torch.cat((preds, next_tokens), dim=1)  # Append to predictions
    
    return preds



# def generate(model, src, max_len, start_symbol, device):
#     model.eval()
#     src = src.to(device)
    
#     with torch.no_grad():
#         encoder_output = model.encoder(model.positional_encoding(model.src_embedding(src)))
        
#         # Initialize the decoder input with the start symbol
#         decoder_input = torch.ones(1, 1).fill_(start_symbol).type_as(src).to(device)
        
#         for i in range(max_len - 1):
#             decoder_output = model.decoder(
#                 model.positional_encoding(model.tgt_embedding(decoder_input)),
#                 encoder_output
#             )
            
#             # Get the next token
#             next_token = decoder_output.argmax(2)[:, -1].unsqueeze(0)
#             decoder_input = torch.cat([decoder_input, next_token], dim=1)
            
#             # Stop if end of sequence token is predicted
#             if next_token.item() == model.tgt_embedding.num_embeddings - 1:  # Assuming last index is <eos>
#                 break
    
#     return decoder_input.squeeze(0)

# def evaluate(model, dataloader, criterion, device, tgt_vocab, max_len):
#     model.eval()
#     epoch_loss = 0
#     all_preds = []
#     all_refs = []

#     scorer_bleu = sacrebleu.metrics.BLEU()
#     rouge_scorer_obj = rouge_scorer.RougeScorer(['rouge1', 'rougeL'], use_stemmer=True)

#     # Assuming '<sos>' is not in the vocabulary, we'll use the index after '<unk>'
#     start_symbol = tgt_vocab.unk_idx + 1

#     with torch.no_grad():
#         for src_batch, tgt_batch in tqdm(dataloader, desc='Evaluating', leave=False):
#             src_batch, tgt_batch = src_batch.to(device), tgt_batch.to(device)

#             # Shift target tokens for teacher forcing
#             tgt_input = tgt_batch[:, :-1]
#             tgt_output = tgt_batch[:, 1:]

#             # Forward pass
#             output = model(src_batch, tgt_input)

#             # Calculate loss
#             loss = criterion(output.contiguous().view(-1, output.size(-1)), tgt_output.contiguous().view(-1))
#             epoch_loss += loss.item()

#             # Generate predictions
#             preds = [generate(model, src.unsqueeze(0), max_len, start_symbol, device) 
#                      for src in src_batch]

#             # Decode token indices to text
#             preds_text = [tgt_vocab.decode(pred.cpu().tolist()) for pred in preds]
#             refs_text = [[tgt_vocab.decode(ref[1:].cpu().tolist())] for ref in tgt_batch]  # Skip the start token

#             all_preds.extend(preds_text)
#             all_refs.extend(refs_text)

#     # Compute BLEU score
#     bleu_score = scorer_bleu.corpus_score(all_preds, all_refs).score

#     # Compute ROUGE score
#     rouge_scores = {key: [] for key in ['rouge1', 'rougeL']}
#     for pred, ref in zip(all_preds, all_refs):
#         scores = rouge_scorer_obj.score(ref[0], pred)
#         for key in rouge_scores:
#             rouge_scores[key].append(scores[key].fmeasure)

#     # Average ROUGE scores
#     rouge_scores = {key: sum(scores) / len(scores) for key, scores in rouge_scores.items()}

#     return epoch_loss / len(dataloader), bleu_score, rouge_scores


def train_epoch(model, dataloader, criterion, optimizer, device, scheduler=None, grad_clip=1.0):
    model.train()
    epoch_loss = 0
    
    # Use tqdm for progress bar
    progress_bar = tqdm(dataloader, desc='Training', leave=False, position=0, ncols=80)
    
    for src_batch, tgt_batch in progress_bar:
        src_batch, tgt_batch = src_batch.to(device), tgt_batch.to(device)
        
        # No need to explicitly call .long() if your data loader is set up correctly
        
        # Shift target tokens for teacher forcing
        tgt_input = tgt_batch[:, :-1]
        tgt_output = tgt_batch[:, 1:]

        # Zero the gradients
        optimizer.zero_grad()

        # Forward pass
        output = model(src_batch, tgt_input)
        
        # Flatten the output and target for cross entropy loss
        output = output.contiguous().view(-1, output.size(-1))
        tgt_output = tgt_output.contiguous().view(-1)

        # Calculate loss
        loss = criterion(output, tgt_output)
        
        # Backward pass and optimize
        loss.backward()
        
        # Gradient clipping
        if grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        
        optimizer.step()

        # Track loss for this epoch
        epoch_loss += loss.item()
        
        # Update progress bar
        progress_bar.set_postfix({'loss': f'{loss.item():.4f}'})
    
    # Step the learning rate scheduler if it's provided
    if scheduler is not None:
        if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
            scheduler.step(epoch_loss)
        else:
            scheduler.step()

    return epoch_loss / len(dataloader)



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
    # optimizer = optim.Adam(model.parameters(), lr=config.learning_rate, weight_decay=1e-6)
    optimizer = optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=1e-2)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=2, gamma=0.1)

    best_valid_loss = float('inf')

    for epoch in range(epochs):
        # Train for one epoch
        train_loss = train_epoch(model, train_loader, criterion, optimizer, device, scheduler)
        print(f'Epoch {epoch+1}/{epochs} Training Loss: {train_loss:.4f}')

        # Validate the model
        tgt_vocab = train_loader.dataset.tgt_vocab
        # valid_loss, bleu, rouge = evaluate(model, dev_loader, criterion, device, tgt_vocab)
        
        valid_loss, bleu, rouge = evaluate(model, dev_loader, criterion, device, tgt_vocab, max_len)
        print(f'Validation Loss: {valid_loss:.4f}, BLEU: {bleu:.4f}, ROUGE-L: {rouge["rougeL"]:.4f}')  # fmeasure

        # Log the results in wandb
        wandb.log({
            "train_loss": train_loss,
            "valid_loss": valid_loss,
            "BLEU": bleu,
            "ROUGE-L": rouge["rougeL"],
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
    
    
    

