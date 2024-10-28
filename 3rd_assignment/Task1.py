import torch
from datasets import load_dataset
from torch.utils.data import DataLoader
from transformers import GPT2Tokenizer, GPT2LMHeadModel
from torch.nn.utils.rnn import pad_sequence
from transformers import BitsAndBytesConfig

scratch_dir = "/scratch/hmnshpl/anlp_data"

# Load CNN/DailyMail dataset (you can use 'test' and 'validation' splits for evaluation)
dataset = load_dataset("cnn_dailymail", "3.0.0", cache_dir=scratch_dir)

# Initialize tokenizer
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")

# Set pad_token as eos_token
tokenizer.pad_token = tokenizer.eos_token
tokenizer.pad_token_id = tokenizer.eos_token_id  # Explicitly set pad_token_id

# Custom collate function to handle batch conversion to PyTorch tensors
def collate_fn(batch):
    input_ids = [torch.tensor(item["input_ids"]) for item in batch]
    labels = [torch.tensor(item["labels"]) for item in batch]

    input_ids = pad_sequence(input_ids,
                batch_first=True, padding_value=tokenizer.pad_token_id)
    labels = pad_sequence(labels,
            batch_first=True, padding_value=-100)  # -100 is often used to ignore padding in loss calculation

    # attention_mask = (input_ids != tokenizer.pad_token_id).long()

    return {
        "input_ids": input_ids,
        "labels": labels,
        # "attention_mask": attention_mask
    }


def preprocess_function(examples):
    inputs = examples['article']  # Articles as inputs
    targets = examples['highlights']  # Highlights as summaries

    # Tokenize inputs and targets
    model_inputs = tokenizer(inputs, max_length=512,
                            truncation=True, padding="max_length",
                            return_tensors="pt")
    labels = tokenizer(targets, max_length=150,
                    truncation=True, padding="max_length",
                    return_tensors="pt").input_ids

    # Replace padding token ids in labels with -100 to ignore padding in loss computation
    labels = torch.where(labels == tokenizer.pad_token_id,
                        -100, labels)
    model_inputs["labels"] = labels
    return model_inputs


# check if GPU is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f'Using {device = }')

# Define the number of soft prompt tokens
num_soft_tokens = 10
soft_prompt_dim = 768  # GPT-2 small's hidden size

# Tokenize the dataset (train and validation splits)
tokenized_datasets = dataset.map(preprocess_function,
                                batched=True,
                                remove_columns=
                                ["article", "highlights", "id"]
                                )

# Convert datasets to PyTorch DataLoader format
train_dataset = tokenized_datasets["train"]
val_dataset = tokenized_datasets["validation"]

# Set up DataLoader
batch_size = 8
train_dataloader = DataLoader(train_dataset, shuffle=True,
                            batch_size=batch_size,
                            collate_fn=collate_fn)
val_dataloader = DataLoader(val_dataset, batch_size=batch_size,
                            collate_fn=collate_fn)


# prompt tuning involves learning soft prompt embeddings that are prepended to the input, while
# keeping the model's weight frozen.

# TODO: Initialize soft prompts

# Initialize learnable soft prompts
soft_prompts = torch.randn((num_soft_tokens, soft_prompt_dim),
                        requires_grad=True, device=device)

quantization_config = BitsAndBytesConfig(load_in_8bit=True)

# Load tokenizer and GPT-2 model
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")

# device_map='auto' caused issue
model = GPT2LMHeadModel.from_pretrained("gpt2",
                        # device_map='auto',
                        device_map={"": 0},
                        quantization_config=quantization_config)  # can't use to device here.
model.config.pad_token_id = tokenizer.pad_token_id
# TODO: Freeze the Model

# Freeze the entire GPT-2 model
for param in model.parameters():
    param.requires_grad = False

# TODO: Prepend Soft prompts

def prepend_prompts(input_ids, soft_prompts):
    '''Prepend soft prompts to input tokens'''
    batch_size = input_ids.shape[0]
    # Repeat the soft prompts for each example in the batch
    soft_prompt_batch = soft_prompts.unsqueeze(0).expand(batch_size, -1, -1)
    # Pass both soft prompts and input tokens to the model
    return torch.cat([soft_prompt_batch, model.transformer.wte(input_ids)], dim=1)

def forward_with_prompts(input_ids, labels=None):
    # Generate attention mask based on input_ids
    attention_mask = (input_ids != tokenizer.pad_token_id).long()
    
    # Prepend soft prompts
    inputs_embeds = prepend_prompts(input_ids, soft_prompts)
    
    # print(f'{input_ids.shape = }, {inputs_embeds.shape = }, {labels.shape = }')
    
    
    # Adjust attention mask to match the new sequence length
    prompt_attention = torch.ones((input_ids.size(0), inputs_embeds.size(1) - input_ids.size(1)), 
                                device=attention_mask.device, dtype=attention_mask.dtype)
    attention_mask = torch.cat([prompt_attention, attention_mask], dim=1)
    
    # Adjust labels if provided
    if labels is not None:
        pad_length = inputs_embeds.size(1) - labels.size(1)
        labels = torch.nn.functional.pad(labels, (0, pad_length), value=-100)
    
    # Ensure consistent data types
    inputs_embeds = inputs_embeds.to(model.dtype)
    attention_mask = attention_mask.to(model.dtype)
    if labels is not None:
        labels = labels.to(torch.long)
    
    # Forward pass
    outputs = model(
        inputs_embeds=inputs_embeds,
        attention_mask=attention_mask,
        labels=labels
    )
    
    return outputs


# TODO: Train only the prompts

# Optimizer for only the soft prompts
optimizer = torch.optim.Adam([soft_prompts], lr=5e-5)

epochs = 10

for epoch in range(epochs):
    
    for batch in train_dataloader:
        input_ids = batch["input_ids"].to(device)
        labels = batch["labels"].to(device)
        

        # Forward pass with soft prompts
        outputs = forward_with_prompts(input_ids, labels=labels)
        loss = outputs.loss

        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    print(f"Epoch {epoch}, Loss: {loss.item()}")

