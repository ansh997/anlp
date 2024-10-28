import torch
import torch.nn as nn
from datasets import load_dataset
from transformers import Trainer, TrainingArguments
from transformers import GPT2LMHeadModel, GPT2Tokenizer
import wandb

scratch_dir = "/scratch/hmnshpl/anlp_data"


# Load the model and tokenizer
model_name = 'gpt2'
tokenizer = GPT2Tokenizer.from_pretrained(model_name)
model = GPT2LMHeadModel.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token  # Set pad_token as eos_token
tokenizer.pad_token_id = tokenizer.eos_token_id  # Explicitly set pad_token_id


class LoRALayer(nn.Module):
    def __init__(self, original_layer, rank):
        super(LoRALayer, self).__init__()
        self.original_layer = original_layer
        self.rank = rank
        
        # Get input and output dimensions
        input_dim = original_layer.weight.shape[1]
        output_dim = original_layer.weight.shape[0]
        
        # Initialize low-rank matrices
        self.lora_a = nn.Parameter(torch.randn(input_dim, rank) * 0.01)  # [input_dim, rank]
        self.lora_b = nn.Parameter(torch.randn(rank, output_dim) * 0.01)  # [rank, output_dim]

    def forward(self, x):
        # Original layer output
        original_output = self.original_layer(x)
        
        # Low-rank adaptation
        lora_output = x @ self.lora_a @ self.lora_b
        return original_output + lora_output


def apply_lora_to_model(model, rank):
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            lora_layer = LoRALayer(module, rank)
            # Replace the original linear layer with the LoRA layer
            setattr(model, name, lora_layer)

# Specify the rank for LoRA
lora_rank = 8  # [4, 8, 16]
apply_lora_to_model(model, lora_rank)

# Freeze the Original Model Parameters
for param in model.parameters():
    param.requires_grad = False

# Unfreeze the LoRA parameters
for name, module in model.named_modules():
    if isinstance(module, LoRALayer):
        for param in module.parameters():
            param.requires_grad = True
            
# Load your dataset (example using a summarization dataset)
dataset = load_dataset('cnn_dailymail', '3.0.0',
                    split='train', cache_dir=scratch_dir)

# Tokenization
def tokenize_function(examples):
    return tokenizer(examples['article'],
                    padding="max_length", truncation=True)

tokenized_dataset = dataset.map(tokenize_function, batched=True)

# Initialize wandb
wandb.init(project="ANLP-A3-Task2", dir=scratch_dir)

# Training arguments
training_args = TrainingArguments(
    output_dir='./results',
    evaluation_strategy='epoch',
    learning_rate=5e-5,
    per_device_train_batch_size=4,
    num_train_epochs=3,
    weight_decay=0.01,
    logging_dir=f'{scratch_dir}/logs',
    report_to="wandb"
)

# Define a Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset,
    eval_dataset=tokenized_dataset,  # Optionally, use a separate eval dataset
)

# Train the model
trainer.train()  # Cuda run error

# Eval the model
trainer.evaluate()


