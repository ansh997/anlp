import math
import torch
from tqdm import tqdm
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader


from neural_language_model.transformer import *


def train_epoch(model, dataloader, optimizer, device):
    model.train()
    total_loss = 0
    total_tokens = 0
    for _, (x, y) in enumerate(dataloader):
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        output = model(x)
        loss = model.calculate_loss(output, y)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip_value)
        optimizer.step()
        total_loss += loss.item() * y.numel()
        total_tokens += y.numel()
    
    avg_loss = total_loss / total_tokens
    perplexity = math.exp(avg_loss)
    return avg_loss, perplexity


def validate(model, dataloader, device):
    model.eval()
    total_loss = 0
    total_tokens = 0
    with torch.no_grad():
        for x, y in dataloader:
            x, y = x.to(device), y.to(device)
            output = model(x)
            loss = model.calculate_loss(output, y)
            total_loss += loss.item() * y.numel()
            total_tokens += y.numel()
    
    avg_loss = total_loss / total_tokens
    perplexity = math.exp(avg_loss)
    return avg_loss, perplexity


def generate_text_with_sampling_and_perplexity(model, start_sequence, vocab, max_length=100, temperature=0.7, top_k=50, top_p=0.9):
    model.eval()
    generated_sequence = start_sequence.copy()
    perplexities = []
    
    with torch.no_grad():
        for _ in range(max_length):
            input_tensor = torch.tensor(vocab.encode(generated_sequence)).unsqueeze(0).to(device)
            output = model(input_tensor)
            next_token_logits = output[0, -1, :] / temperature
            
            # Top-k sampling
            top_k_logits, top_k_indices = torch.topk(next_token_logits, top_k)
            
            # Top-p (nucleus) sampling
            sorted_logits, sorted_indices = torch.sort(top_k_logits, descending=True)
            cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
            sorted_indices_to_remove = cumulative_probs > top_p
            sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
            sorted_indices_to_remove[..., 0] = 0
            indices_to_remove = sorted_indices[sorted_indices_to_remove]
            top_k_logits[indices_to_remove] = float('-inf')
            
            # Sample from the filtered distribution
            probs = F.softmax(top_k_logits, dim=-1)
            next_token_idx = top_k_indices[torch.multinomial(probs, 1).item()].item()
            next_token = vocab.itos[next_token_idx]
            
            # Calculate perplexity for the generated token
            token_prob = probs[top_k_indices == next_token_idx].item()  # Probability of the sampled token
            perplexity = math.exp(-math.log(token_prob)) if token_prob > 0 else float('inf')
            perplexities.append(perplexity)
            
            generated_sequence.append(next_token)
            print(f"Generated Token: {next_token}, Perplexity: {perplexity:.4f}")
            
            # Break if end-of-sequence token is generated or max length is reached
            if next_token == "<eos>" or len(generated_sequence) >= max_length:
                break
    
    return " ".join(generated_sequence), perplexities






if __name__ == '__main__':
    
    corpus = get_auguste_maquet_corpus(file_path=data_filepath)
    tokens = tokenize(corpus)
    vocab = Vocabulary(tokens)
    
    # Training setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    seq_length = 128  # Increased sequence length
    batch_size = 64   # Increased batch size
    d_model = 512     # Increased model dimension
    nhead = 8
    num_layers = 6    # Increased number of layers
    dim_feedforward = 2048
    lr = 0.0001       # Decreased learning rate
    epochs = 10       # Increased number of epochs

    dataset = TextDataset(corpus, vocab, seq_length)
    train_size = int(0.9 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])

    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size)

    model = TransformerDecoder(len(vocab), d_model, nhead, num_layers, dim_feedforward).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)
    
    
    # Gradient clipping
    clip_value = 1.0
    
    # Training loop
    for epoch in tqdm(range(epochs), desc='Training'):
        train_loss, train_perplexity = train_epoch(model, train_dataloader, optimizer, device)
        val_loss, val_perplexity = validate(model, val_dataloader, device)
        scheduler.step()
        
        print(f"Epoch {epoch+1}/{epochs}")
        print(f"Train Loss: {train_loss:.4f}, Train Perplexity: {train_perplexity:.4f}")
        print(f"Val Loss: {val_loss:.4f}, Val Perplexity: {val_perplexity:.4f}")
    
    start_sequence = ["the", "count", "of"]
    generated_text, perplexities = generate_text_with_sampling_and_perplexity(model, start_sequence, vocab, temperature=0.7, top_k=50, top_p=0.9)

    print("Generated Text:")
    print(generated_text)
    print("Perplexities:", perplexities)
        
    
    
    
    