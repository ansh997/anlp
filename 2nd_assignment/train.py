import torch
import torch.nn as nn
import torch.optim as optim
from encoder import Encoder
from decoder import Decoder
from Old_dataset import get_dataloaders
from utils import PositionalEncoding
from tqdm.auto import tqdm
import warnings
import numpy as np
import sacrebleu
from rouge_score import rouge_scorer


def clip_gradients(model: nn.Module, max_norm: float = 1.0) -> None:
    """
    Clips gradients of the model's parameters.
    """
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)

def init_weights(m):
    if isinstance(m, nn.Linear):
        torch.nn.init.xavier_uniform_(m.weight)
        
def training_diagnostics(model, loss, optimizer):
    """
    Print diagnostic information about the current state of training.
    """
    print("\nTraining Diagnostics:")
    
    # Loss value
    print(f"Current loss: {loss.item():.4f}")
    
    # Gradient statistics
    grad_norms = []
    for name, param in model.named_parameters():
        if param.grad is not None:
            grad_norm = param.grad.norm().item()
            grad_norms.append(grad_norm)
            if np.isnan(grad_norm) or np.isinf(grad_norm):
                print(f"Warning: NaN or Inf gradient in {name}")
    
    if grad_norms:
        print(f"Gradient norm - Mean: {np.mean(grad_norms):.4f}, Max: {np.max(grad_norms):.4f}, Min: {np.min(grad_norms):.4f}")
    
    # Parameter statistics
    for name, param in model.named_parameters():
        if torch.isnan(param).any() or torch.isinf(param).any():
            print(f"Warning: NaN or Inf value in parameters of {name}")
    
    # Learning rate
    for param_group in optimizer.param_groups:
        print(f"Current learning rate: {param_group['lr']:.6f}")
    
    print("\n")

class Transformer(nn.Module):
    def __init__(self, encoder, decoder, model_dim, src_vocab_size, tgt_vocab_size, max_len):
        super(Transformer, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.src_embedding = nn.Embedding(src_vocab_size, model_dim)
        self.tgt_embedding = nn.Embedding(tgt_vocab_size, model_dim)
        self.positional_encoding = PositionalEncoding(model_dim, max_len)
        
    def forward(self, src, tgt):
        # Apply embeddings
        src_emb = self.src_embedding(src)
        tgt_emb = self.tgt_embedding(tgt)
        
        assert src_emb.device == tgt_emb.device, "Source and target embeddings are not on the same device"

        # Apply positional encoding
        src_emb = self.positional_encoding(src_emb)
        tgt_emb = self.positional_encoding(tgt_emb)
        
        # print("\ttransformer: Shape of src_emb:", src_emb.shape, "Shape of tgt_emb:", tgt_emb.shape)

        # Pass through encoder and decoder
        enc_output = self.encoder(torch.tensor(src_emb, dtype=torch.long))
        output = self.decoder(tgt_emb, enc_output)
        return output

# Training function
def old_train_epoch(model, dataloader, criterion, optimizer, device, scheduler=None):
    model.train()
    epoch_loss = 0
    

    for src_batch, tgt_batch in tqdm(dataloader, desc='Training', leave=False, position=0, ncols=80):
        src_batch, tgt_batch = src_batch.to(device), tgt_batch.to(device)
        
        src_batch = src_batch.long()
        tgt_batch = tgt_batch.long()

        # Shift target tokens for teacher forcing
        tgt_input = tgt_batch[:, :-1]
        tgt_output = tgt_batch[:, 1:]

        # Zero the gradients
        optimizer.zero_grad()

        # Forward pass
        # print("\ttrain_epoch: Shape of src_batch:", src_batch.shape, "Shape of tgt_input:", tgt_input.shape)
        output = model(src_batch, tgt_input)  # are on same device
        
        # print('\nJust after model output: ', output.argmax(dim=-1).shape, output.argmax(dim=-1), output.shape, tgt_output.shape)
        # exit()
        
        
        
        
        # print(f'Output shape before flattening: {output.shape}, tgt_output shape before flattening: {tgt_output.shape}')
        
        # print('\ttrain_epoch: Shape of output:', output.shape, 'Shape of tgt_output:', tgt_output.shape)

        # Flatten the output and target for cross entropy loss
        output = output.permute(0, 2, 1).contiguous()
        # output = output.view(-1, output.shape[-1])
        # tgt_output = tgt_output.reshape(-1)
        # print('*'*80)
        # print(output.argmax(dim=-1).shape, output.shape,  len(output.argmax(dim=-1)))
        # print(tgt_output.shape, tgt_output.argmax(dim=-1))
        # print('*'*80)
        # exit()

        # Calculate loss
        # print("Output shape:", output.shape)  # Expected: [64, 22586, 99]
        # print("Target shape:", tgt_output.shape)  # Expected: [64, 99]
        
        # exit()

        loss = criterion(output, tgt_output)
        
        # Backward pass and optimize
        loss.backward()
        # training_diagnostics(model, loss, optimizer)
        clip_gradients(model, max_norm=1.0)
        optimizer.step()

        # Track loss for this epoch
        epoch_loss += loss.item()
        
    if scheduler is not None:
        scheduler.step()

    return epoch_loss / len(dataloader)


def train_epoch(model, dataloader, criterion, optimizer, device, scheduler=None):
    model.train()
    epoch_loss = 0
    total_tokens = 0
    correct_predictions = 0

    for src_batch, tgt_batch in tqdm(dataloader, desc='Training', leave=False, position=0, ncols=80):
        src_batch, tgt_batch = src_batch.to(device), tgt_batch.to(device)
        
        src_batch = src_batch.long()
        tgt_batch = tgt_batch.long()

        # Shift target tokens for teacher forcing
        tgt_input = tgt_batch[:, :-1]
        tgt_output = tgt_batch[:, 1:]

        # Zero the gradients
        optimizer.zero_grad()

        # Forward pass
        output = model(src_batch, tgt_input)
        
        # Reshape output and target for loss calculation
        output = output.view(-1, output.size(-1))
        tgt_output = tgt_output.contiguous().view(-1)

        # Calculate loss
        loss = criterion(output, tgt_output)
        
        # Backward pass and optimize
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        # Track loss and accuracy
        epoch_loss += loss.item()
        # total_tokens += tgt_output.ne(model.pad_token_id).sum().item()
        correct_predictions += (output.argmax(dim=-1) == tgt_output).sum().item()

        # Debug print
        if total_tokens > 0:
            print(f"Batch loss: {loss.item():.4f}, Accuracy: {correct_predictions / total_tokens:.4f}")
            print(f"Sample prediction: {output.argmax(dim=-1)[:10]}")
            print(f"Sample target: {tgt_output[:10]}")

    if scheduler is not None:
        scheduler.step()

    avg_loss = epoch_loss / len(dataloader)
    # accuracy = correct_predictions / total_tokens if total_tokens > 0 else 0
    # print(f'{accuracy=}')

    return avg_loss


# Validation function
def _evaluate(model, dataloader, criterion, device):
    model.eval()
    epoch_loss = 0

    with torch.no_grad():
        for src_batch, tgt_batch in dataloader:
            src_batch, tgt_batch = src_batch.to(device), tgt_batch.to(device)

            # Shift target tokens
            tgt_input = tgt_batch[:, :-1]
            tgt_output = tgt_batch[:, 1:]

            # Forward pass
            output = model(src_batch, tgt_input)

            # Flatten the output and target for cross entropy loss
            output = output.view(-1, output.shape[-1])
            tgt_output = tgt_output.reshape(-1)

            # Calculate loss
            loss = criterion(output, tgt_output)
            epoch_loss += loss.item()

    return epoch_loss / len(dataloader)


def evaluate(model, dataloader, criterion, device, tgt_vocab):
    model.eval()
    epoch_loss = 0
    all_preds = []
    all_refs = []

    scorer_bleu = sacrebleu.metrics.BLEU()
    rouge_scorer_obj = rouge_scorer.RougeScorer(['rouge1', 'rougeL'], use_stemmer=True)

    with torch.no_grad():
        for src_batch, tgt_batch in dataloader:
            src_batch, tgt_batch = src_batch.to(device), tgt_batch.to(device)

            # Shift target tokens for teacher forcing
            tgt_input = tgt_batch[:, :-1]
            tgt_output = tgt_batch[:, 1:]

            # Forward pass
            output = model(src_batch, tgt_input)

            # Flatten the output and target for cross-entropy loss
            output = output.view(-1, output.shape[-1])
            # print(f'{output.shape=}')
            tgt_output = tgt_output.reshape(-1)
            # print(f'{tgt_output.shape=}')

            # Calculate loss
            loss = criterion(output, tgt_output)
            epoch_loss += loss.item()

            # Convert model output to predicted token indices
            # print(output.argmax(dim=-1).shape, output.argmax(dim=-1), src_batch.size(0))
            preds = output.argmax(dim=-1).view(src_batch.size(0), -1).tolist()
            refs = tgt_output.view(src_batch.size(0), -1).tolist()

            # Decode token indices to text
            # The code snippet you provided is printing and displaying the contents of the `preds`
            # variable. Here's a breakdown of what each line does:
            print(len(preds))
            print('*'*80)
            print('\n', preds)
            print('*'*80)
            # exit()
            preds_text = [tgt_vocab.decode(pred) for pred in preds]
            refs_text = [[tgt_vocab.decode(ref)] for ref in refs]  # BLEU expects a list of reference lists

            all_preds.extend(preds_text)
            all_refs.extend(refs_text)

    # Compute BLEU score
    bleu_score = scorer_bleu.corpus_score(all_preds, all_refs).score

    # Compute ROUGE score
    refs_text_concat = "\n".join([" ".join(ref) for ref in all_refs])
    preds_text_concat = "\n".join(all_preds)
    
    print('refs_text_concat:', refs_text_concat)
    print('preds_text_concat:', preds_text_concat)
    exit()
    rouge_scores = rouge_scorer_obj.score(refs_text_concat, preds_text_concat)

    return epoch_loss / len(dataloader), bleu_score, rouge_scores



# Main training loop
def train_model(scratch_location, src_lang, tgt_lang, epochs=10, batch_size=64, max_len=100, learning_rate=0.0001, tokenizer_type='simple'):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')

    # Get data loaders
    print('Loading data...', end='\r')
    train_loader, dev_loader, _ = get_dataloaders(scratch_location, src_lang, tgt_lang, batch_size, max_len, tokenizer=tokenizer_type)
    print('Loaded data successfully.')

    print('Instantiate the encoder and decoder', end='\r')
    src_vocab_size = train_loader.dataset.src_vocab.__len__()
    tgt_vocab_size = train_loader.dataset.tgt_vocab.__len__()
    model_dim = 512  # Embedding size and model hidden size

    encoder = Encoder(src_vocab_size, model_dim, num_heads=8, num_layers=6, d_ff=2048, max_seq_len=max_len)
    decoder = Decoder(tgt_vocab_size, model_dim, num_heads=8, num_layers=6, d_ff=2048, max_seq_len=max_len)
    
    print(f'Instantiated the encoder and decoder')

    # Full transformer model
    model = Transformer(encoder, decoder, model_dim, src_vocab_size, tgt_vocab_size, max_len).to(device)  # model is set to device
    
    model.apply(init_weights)  # helps
    
    print('Training the model...', end='\r')

    # criterion = nn.CrossEntropyLoss(ignore_index=train_loader.dataset.tgt_vocab.pad_idx)
    criterion = nn.CrossEntropyLoss(ignore_index=train_loader.dataset.tgt_vocab.pad_idx, label_smoothing=0.1).to(device)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-6)
    # scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=1e-3, steps_per_epoch=len(train_loader), epochs=5)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=2, gamma=0.1)

    best_valid_loss = float('inf')

    for epoch in range(epochs):

        # Train for one epoch
        train_loss = train_epoch(model, train_loader, criterion, optimizer, device, scheduler)
        print(f'Epoch {epoch+1}/{epochs}', end = ' ')
        print(f'Training Loss: {train_loss:.4f}', end = ' ')

        # Validate the model
        # valid_loss = evaluate(model, dev_loader, criterion, device)
        # print(f'Validation Loss: {valid_loss:.4f}')
        tgt_vocab = train_loader.dataset.tgt_vocab
        # print('0: ', tgt_vocab.decode('0'), '1: ', tgt_vocab.decode(1), '2: ', tgt_vocab.decode(2))
        # exit()
        valid_loss, bleu, rouge = evaluate(model, dev_loader, criterion, device, tgt_vocab)
        print(f'Validation Loss: {valid_loss:.4f}, BLEU: {bleu:.4f}, ROUGE-L: {rouge["rougeL"].fmeasure:.4f}')
        if epoch == 5:
            exit()


        # Save the model if it has the best validation loss so far
        if valid_loss < best_valid_loss:
            best_valid_loss = valid_loss
            torch.save(model.state_dict(), 'transformer.pt')

    print('Training complete!')

if __name__ == '__main__':
    warnings.filterwarnings("ignore")
    scratch_location = "/scratch/hmnshpl/anlp_data/ted-talks-corpus"
    src_lang = "en"
    tgt_lang = "fr"
    train_model(scratch_location, src_lang, tgt_lang, epochs=5, batch_size=32, max_len=100, learning_rate=0.000001, tokenizer_type='simple')
    # TODO: Add BLEU, ROUGE Evaluation scores
