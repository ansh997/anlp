import torch
from tqdm import tqdm
from datasets import load_dataset
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence
from peft import LoraConfig, get_peft_model
from transformers import AutoTokenizer, AutoModelForCausalLM, logging
import sacrebleu
from rouge_score import rouge_scorer

import warnings

# Ignore all warnings
warnings.filterwarnings("ignore")
logging.set_verbosity_error()

def generate_summary(model, input_ids, attention_mask):
    outputs = model.generate(
        input_ids=input_ids,
        attention_mask=attention_mask,
        max_new_tokens=150,
        min_length=40,
        num_beams=4,
        early_stopping=True,
        no_repeat_ngram_size=2,
        pad_token_id=tokenizer.pad_token_id,
        bos_token_id=tokenizer.bos_token_id,
        eos_token_id=tokenizer.eos_token_id
    )
    return outputs

def preprocess_function(examples):
    # Combine input and target text with a separator
    model_inputs = []
    for article, highlight in zip(examples['article'], examples['highlights']):
        # Format: "<|endoftext|>Article: {article} Summary: {highlight}<|endoftext|>"
        combined_text = f"{tokenizer.eos_token}Article: {article} Summary: {highlight}{tokenizer.eos_token}"
        model_inputs.append(combined_text)
    
    # Tokenize with padding and truncation
    encodings = tokenizer(
        model_inputs,
        max_length=512,
        truncation=True,
        padding=True
    )
    
    # Convert to appropriate format
    input_ids = encodings['input_ids']
    attention_mask = encodings['attention_mask']
    labels = input_ids.copy()
    
    # Convert to dictionary format
    return {
        'input_ids': input_ids,
        'attention_mask': attention_mask,
        'labels': labels
    }

def collate_fn(batch):
    # Extract each component
    input_ids = [torch.tensor(item['input_ids']) for item in batch]
    attention_mask = [torch.tensor(item['attention_mask']) for item in batch]
    labels = [torch.tensor(item['labels']) for item in batch]
    
    # Pad sequences
    input_ids = pad_sequence(input_ids, batch_first=True, padding_value=tokenizer.pad_token_id)
    attention_mask = pad_sequence(attention_mask, batch_first=True, padding_value=0)
    labels = pad_sequence(labels, batch_first=True, padding_value=-100)
    
    return {
        'input_ids': input_ids,
        'attention_mask': attention_mask,
        'labels': labels
    }

def evaluate(model, val_dataloader, device):
    model.eval()
    total_val_loss = 0
    generated_summaries = []
    reference_summaries = []
    rouge_scores = {'rouge1': 0, 'rouge2': 0, 'rougeL': 0}
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
    num_samples = 0
    
    with torch.no_grad():
        for batch in tqdm(val_dataloader, desc="Validating"):
            input_ids = batch["input_ids"].to(device)
            labels = batch["labels"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            
            # Calculate loss
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels
            )
            loss = outputs.loss
            total_val_loss += loss.item()
            
            # Generate summaries for BLEU and ROUGE
            generated_ids = generate_summary(model, input_ids, attention_mask)
            
            # Convert ids to text
            batch_generated = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
            batch_references = tokenizer.batch_decode(labels, skip_special_tokens=True)
            
            # Calculate ROUGE scores for this batch
            for gen, ref in zip(batch_generated, batch_references):
                scores = scorer.score(ref, gen)
                rouge_scores['rouge1'] += scores['rouge1'].fmeasure
                rouge_scores['rouge2'] += scores['rouge2'].fmeasure
                rouge_scores['rougeL'] += scores['rougeL'].fmeasure
                num_samples += 1
            
            # Collect summaries for BLEU calculation
            generated_summaries.extend(batch_generated)
            reference_summaries.extend(batch_references)
    
    # Calculate BLEU score using sacrebleu
    bleu = sacrebleu.corpus_bleu(generated_summaries, [reference_summaries])
    
    # Calculate averages
    avg_val_loss = total_val_loss / len(val_dataloader)
    avg_rouge = {k: v / num_samples for k, v in rouge_scores.items()}
    
    return {
        'loss': avg_val_loss,
        'bleu': bleu.score,
        'rouge': avg_rouge
    }
    
    
# Load tokenizer
# Initialize model and tokenizer
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = AutoModelForCausalLM.from_pretrained("gpt2").to(device)
tokenizer = AutoTokenizer.from_pretrained("gpt2", padding_side='left')
tokenizer.padding_side = "left"
tokenizer.pad_token = tokenizer.eos_token
tokenizer.pad_token_id = tokenizer.eos_token_id

# Freeze model weights
for param in model.parameters():
    param.requires_grad = False

# Configure LoRA
config = LoraConfig(
    r=16,
    lora_alpha=32,
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM"
)

model = get_peft_model(model, config).to(device)

# Load and preprocess dataset
scratch_dir = "/scratch/hmnshpl/anlp_data"
dataset = load_dataset('cnn_dailymail', '3.0.0', cache_dir=scratch_dir)
test_dataset = dataset['test'].select(range(100))  # Use a subset of the test dataset

test_dataset = test_dataset.map(
    preprocess_function,
    batched=True,
    remove_columns=["article", "highlights", "id"]
)

# Create the DataLoader
batch_size = 8  # Adjust as necessary
test_dataloader = DataLoader(
    test_dataset,
    batch_size=batch_size,
    collate_fn=collate_fn,
    shuffle=False
)

# Move model to appropriate device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

# Evaluate the model on the test data
test_results = evaluate(model, test_dataloader, device)

# Print the results
print(f"Test Loss: {test_results['loss']}")
print(f"Test BLEU Score: {test_results['bleu']}")
print(f"Test ROUGE Scores: {test_results['rouge']}")