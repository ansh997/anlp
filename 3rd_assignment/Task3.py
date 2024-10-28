import torch
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from transformers import GPT2LMHeadModel, GPT2Tokenizer, get_linear_schedule_with_warmup
from datasets import load_dataset
from tqdm import tqdm

class SummarizationDataset(Dataset):
    def __init__(self, texts, summaries, tokenizer, max_length=1024):
        self.tokenizer = tokenizer
        self.texts = texts
        self.summaries = summaries
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        summary = self.summaries[idx]
        
        # Create input with special tokens
        input_text = f"Summarize: {text} TL;DR: {summary}"
        
        # Tokenize and prepare input
        encodings = self.tokenizer(input_text,
                                truncation=True,
                                max_length=self.max_length,
                                padding='max_length',
                                return_tensors='pt')
        
        input_ids = encodings['input_ids'].squeeze()
        attention_mask = encodings['attention_mask'].squeeze()
        
        # Create labels (shift input_ids left by 1)
        labels = input_ids.clone()
        labels[:-1] = input_ids[1:]
        labels[-1] = -100  # Ignore last token for loss calculation
        
        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'labels': labels
        }

def prepare_model_for_last_layer_training(model):
    """Freeze all layers except the last few layers"""
    # Freeze all parameters
    for param in model.parameters():
        param.requires_grad = False
    
    # Unfreeze the last transformer block
    for param in model.transformer.h[-1].parameters():
        param.requires_grad = True
    
    # Unfreeze the output layer (LM head)
    for param in model.lm_head.parameters():
        param.requires_grad = True
    
    return model

def train_model(model, train_dataloader, val_dataloader, device, num_epochs=10):
    # Using PyTorch's AdamW implementation
    optimizer = AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=1e-4,
        weight_decay=0.01,
        betas=(0.9, 0.999),
        eps=1e-8
    )
    
    # Create scheduler with warmup
    num_training_steps = len(train_dataloader) * num_epochs
    num_warmup_steps = num_training_steps // 10
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=num_warmup_steps,
        num_training_steps=num_training_steps
    )
    
    # Training loop
    best_val_loss = float('inf')
    
    for epoch in range(num_epochs):
        model.train()
        total_train_loss = 0
        
        for batch in tqdm(train_dataloader, desc=f'Epoch {epoch + 1}/{num_epochs}'):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels
            )
            
            loss = outputs.loss
            total_train_loss += loss.item()
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
        
        avg_train_loss = total_train_loss / len(train_dataloader)
        
        # Validation
        model.eval()
        total_val_loss = 0
        
        with torch.no_grad():
            for batch in tqdm(val_dataloader, desc='Validation'):
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                labels = batch['labels'].to(device)
                
                outputs = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels
                )
                
                total_val_loss += outputs.loss.item()
        
        avg_val_loss = total_val_loss / len(val_dataloader)
        
        print(f'Epoch {epoch + 1}:')
        print(f'Average training loss: {avg_train_loss:.4f}')
        print(f'Average validation loss: {avg_val_loss:.4f}')
        
        # Save best model
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': best_val_loss,
            }, 'finetune_best_model.pt')

def main():
    # Load dataset
    dataset = load_dataset('cnn_dailymail', '3.0.0')
    
    # Initialize tokenizer and model
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    tokenizer.pad_token = tokenizer.eos_token
    model = GPT2LMHeadModel.from_pretrained('gpt2')
    
    # Prepare model for last-layer training
    model = prepare_model_for_last_layer_training(model)
    
    # Create datasets
    train_dataset = SummarizationDataset(
        dataset['train']['article'][:30000],  # Limit size for memory constraints
        dataset['train']['highlights'][:30000],
        tokenizer
    )
    
    val_dataset = SummarizationDataset(
        dataset['validation']['article'][:10000],
        dataset['validation']['highlights'][:10000],
        tokenizer
    )
    
    # Create dataloaders
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=4,  # Small batch size for memory efficiency
        shuffle=True
    )
    
    val_dataloader = DataLoader(
        val_dataset,
        batch_size=4,
        shuffle=False
    )
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    
    # Train model
    train_model(model, train_dataloader, val_dataloader, device)

if __name__ == "__main__":
    main()