import os
import torch
from datasets import load_dataset, load_from_disk
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from transformers import GPT2Tokenizer, GPT2LMHeadModel, BitsAndBytesConfig, logging
from torch.nn.utils.rnn import pad_sequence
from rouge_score import rouge_scorer
from torch import autocast
import warnings

# Ignore all warnings
warnings.filterwarnings("ignore")
logging.set_verbosity_error()

# Define directories and settings
scratch_dir = "/scratch/hmnshpl/anlp_data"

# Cache directory
cache_dir = "/scratch/hmnshpl/anlp_data/tokenized_data"
batch_size = 8
num_soft_tokens = 10
soft_prompt_dim = 768  # GPT-2 small's hidden size
epochs = 10
lr = 5e-5

print('Loading Dataset', end=' \r')
# Load CNN/DailyMail dataset (use 'test' and 'validation' splits for evaluation)
dataset = load_dataset("cnn_dailymail", "3.0.0", cache_dir=scratch_dir)

# dataset["train"] = dataset["train"].select(range(1000))  # Use a subset of the training data 
# dataset["validation"] = dataset["validation"].select(range(100))  # Use a subset of the validation data  
# dataset["test"] = dataset["test"].select(range(100))  # Use a subset of the validation data  

print(f'Loaded Dataset       ')

dataset["train"] = dataset["train"].select(range(int(len(dataset["train"]) * 0.1)))
dataset["validation"] = dataset["validation"].select(range(int(len(dataset["validation"]) * 0.1)))
dataset["test"] = dataset["test"].select(range(int(len(dataset["test"]) * 0.1)))
print(f"Using 10% of data - Train: {len(dataset['train'])} samples, Val: {len(dataset['validation'])} samples")



# Initialize tokenizer
GPT2Tokenizer.padding_side = "left"
tokenizer = GPT2Tokenizer.from_pretrained("gpt2", padding_side='left')
tokenizer.padding_side = "left"  # Set padding to the left
tokenizer.pad_token = tokenizer.eos_token  # Set pad_token as eos_token
tokenizer.pad_token_id = tokenizer.eos_token_id  # Explicitly set pad_token_id

# Custom collate function to handle batch conversion to PyTorch tensors
def collate_fn(batch):
    input_ids = [torch.tensor(item["input_ids"]) for item in batch]
    labels = [torch.tensor(item["labels"]) for item in batch]

    input_ids = pad_sequence(input_ids, batch_first=True,
                padding_value=tokenizer.pad_token_id)
    labels = pad_sequence(labels, batch_first=True, padding_value=-100)  # -100 to ignore padding in loss calculation
    
    attention_mask = (input_ids != tokenizer.pad_token_id).long()

    return {"input_ids": input_ids, "labels": labels, "attention_mask": attention_mask}

# Preprocessing function
def preprocess_function(examples):
    inputs = examples['article']
    targets = examples['highlights']

    model_inputs = tokenizer(inputs, max_length=512,
                    truncation=True, padding="max_length",
                    return_tensors="pt")
    labels = tokenizer(targets, max_length=150,
            truncation=True, padding="max_length",
            return_tensors="pt").input_ids

    # Replace padding token ids in labels with -100 to ignore padding in loss computation
    labels = torch.where(labels == tokenizer.pad_token_id, -100, labels)
    model_inputs["labels"] = labels
    return model_inputs

def validate_model(model, val_dataloader):
    model.eval()
    total_loss = 0
    rouge_scorer_fn = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
    total_rouge1, total_rouge2, total_rougeL = 0, 0, 0
    # num_batches = 0
    num_instances = 0
    with torch.no_grad():
        for batch in val_dataloader:
            input_ids = batch["input_ids"].to(device)
            labels = batch["labels"].to(device)
            

            # Forward pass with soft prompts
            outputs = forward_with_prompts(input_ids, labels=labels)
            loss = outputs.loss
            total_loss += loss.item()
            
            # Define attention mask and set pad_token_id
            attention_mask = batch["attention_mask"].to(device)
            pad_token_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
            tokenizer.padding_side = "left"  # Set padding to the left
            

            # Decoding predicted and reference sequences for ROUGE scoring
            predictions = model.generate(input_ids,
                                        attention_mask=attention_mask,
                                        max_new_tokens=50,
                                        num_beams=2, no_repeat_ngram_size=2, pad_token_id=pad_token_id)
            # pred_str = tokenizer.batch_decode(predictions, skip_special_tokens=True)
            # label_str = tokenizer.batch_decode(labels, skip_special_tokens=True)
            # Convert labels to a list and check for invalid tokens
            labels_list = labels.tolist()

            # Filter out invalid tokens (None or -100 if using masked tokens)
            filtered_labels = [[token for token in label if token is not None and token != -100] for label in labels_list]

            # Decode the labels
            label_str = [tokenizer.decode(label, skip_special_tokens=True) for label in filtered_labels if label]

            pred_str = [tokenizer.decode(p, skip_special_tokens=True) for p in predictions if p is not None]
            # label_str = [tokenizer.decode(l, skip_special_tokens=True) for l in labels if l is not None]

            
            # Compute ROUGE score for each batch
            for pred, ref in zip(pred_str, label_str):
                rouge_scores = rouge_scorer_fn.score(ref, pred)
                total_rouge1 += rouge_scores['rouge1'].fmeasure
                total_rouge2 += rouge_scores['rouge2'].fmeasure
                total_rougeL += rouge_scores['rougeL'].fmeasure
                num_instances+=1

            # num_batches += 1

    avg_loss = total_loss / len(val_dataloader)
    avg_rouge1 = total_rouge1 / num_instances
    avg_rouge2 = total_rouge2 / num_instances
    avg_rougeL = total_rougeL / num_instances

    return avg_loss, avg_rouge1, avg_rouge2, avg_rougeL

# Check if GPU is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f'Using {device}')

# Tokenize dataset
if os.path.exists(os.path.join(cache_dir, "tokenized_dataset")):
    tokenized_datasets = load_from_disk(os.path.join(cache_dir, "tokenized_dataset"))
    print("Loaded tokenized dataset from cache.")
else:
    tokenized_datasets = dataset.map(preprocess_function,
                    batched=True,
                    remove_columns=["article", "highlights", "id"])
    tokenized_datasets.save_to_disk(os.path.join(cache_dir, "tokenized_dataset"))
    print("Tokenized dataset saved to cache.")


# Prepare DataLoaders
train_dataset = tokenized_datasets["train"]
val_dataset = tokenized_datasets["validation"]
test_dataset = tokenized_datasets["test"]

train_dataloader = DataLoader(train_dataset, shuffle=True, batch_size=batch_size, collate_fn=collate_fn)
val_dataloader = DataLoader(val_dataset, batch_size=batch_size, collate_fn=collate_fn)
test_dataloader = DataLoader(test_dataset, batch_size=batch_size, collate_fn=collate_fn, shuffle=False)

# Check number of batches
# print("Number of batches in val_dataloader:", len(val_dataloader))
# print("Number of batches in test_dataloader:", len(test_dataloader))

# Check dataset sizes (total number of samples)
# print("Number of samples in val_dataloader:", len(val_dataloader.dataset))
# print("Number of samples in test_dataloader:", len(test_dataloader.dataset))

# Initialize soft prompts
soft_prompts = torch.randn((num_soft_tokens, soft_prompt_dim), requires_grad=True, device=device)

# Load GPT-2 model with quantization (8-bit loading)
quantization_config = BitsAndBytesConfig(load_in_8bit=True)
model = GPT2LMHeadModel.from_pretrained("gpt2", device_map='auto',
                                        # quantization_config=quantization_config
                                        )
model.config.pad_token_id = tokenizer.pad_token_id

# Freeze the model's parameters
for param in model.parameters():
    param.requires_grad = False

# Function to prepend soft prompts
def prepend_prompts(input_ids, soft_prompts):
    batch_size = input_ids.shape[0]
    soft_prompt_batch = soft_prompts.unsqueeze(0).expand(batch_size, -1, -1)
    return torch.cat([soft_prompt_batch, model.transformer.wte(input_ids)], dim=1)

# Forward pass with prompts
def forward_with_prompts(input_ids, labels=None):
    attention_mask = (input_ids != tokenizer.pad_token_id).long()
    
    # Prepend soft prompts
    inputs_embeds = prepend_prompts(input_ids, soft_prompts)
    
    # Adjust attention mask
    prompt_attention = torch.ones((input_ids.size(0), inputs_embeds.size(1) - input_ids.size(1)), 
                        device=attention_mask.device, dtype=attention_mask.dtype)
    attention_mask = torch.cat([prompt_attention, attention_mask], dim=1)
    
    # Adjust labels if provided
    if labels is not None:
        pad_length = inputs_embeds.size(1) - labels.size(1)
        labels = torch.nn.functional.pad(labels, (0, pad_length), value=-100)
        
    # Ensure consistent data types
    inputs_embeds = inputs_embeds.to(torch.float32)  # Ensure inputs_embeds is Float
    attention_mask = attention_mask.to(torch.float32)  # Ensure attention_mask is Float
    if labels is not None:
        labels = labels.to(torch.long)  # Ensure labels are Long
    
    # Forward pass through the model
    outputs = model(inputs_embeds=inputs_embeds, attention_mask=attention_mask, labels=labels)
    return outputs

# Optimizer for soft prompts
optimizer = torch.optim.Adam([soft_prompts], lr=lr)
best_val_loss = float('inf')

# Training loop
for epoch in range(epochs):
    model.train()
    total_train_loss = 0
    for idx, batch in tqdm(enumerate(train_dataloader), desc=f"Training epoch: {epoch}/{epochs}", leave=False):
        input_ids = batch["input_ids"].to(device)
        labels = batch["labels"].to(device)

        # Forward pass with soft prompts
        with autocast(str(device)):
            outputs = forward_with_prompts(input_ids, labels=labels)
            loss = outputs.loss
            total_train_loss += loss.item()
        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    
    # Val
    avg_val_loss, avg_rouge1, avg_rouge2, avg_rougeL = validate_model(model, val_dataloader)
    
    # Save best model
    if avg_val_loss < best_val_loss:
        best_val_loss = avg_val_loss
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': best_val_loss,
        }, 'prompt_tune_best_model.pt')
    
    avg_train_loss = total_train_loss / len(train_dataloader)

    print(f"Epoch {epoch + 1}/{epochs}, Loss: {avg_train_loss}", end=', ')
    print(f"Validation Loss: {avg_val_loss}", end=', ')
    print(f"ROUGE-1: {avg_rouge1}, ROUGE-2: {avg_rouge2}, ROUGE-L: {avg_rougeL}")


print(' '*100)
print('*'*100)
avg_test_loss, avg_rouge1, avg_rouge2, avg_rougeL = validate_model(model, test_dataloader)
print(f"Test Loss: {avg_test_loss}", end=', ')
print(f"ROUGE-1: {avg_rouge1}, ROUGE-2: {avg_rouge2}, ROUGE-L: {avg_rougeL}")
print(' '*100)
print('*'*100)