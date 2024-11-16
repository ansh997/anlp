import warnings
warnings.filterwarnings("ignore")
import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from tqdm import tqdm
import numpy as np
import time
from torch.nn import CrossEntropyLoss
from helper import print_model_info, replace_linear_with_target_and_quantize, W8A16LinearLayer

def calculate_perplexity(model, tokenizer, eval_dataset, max_length=512, batch_size=4):
    """
    Calculate perplexity on the evaluation dataset
    """
    model.eval()
    device = next(model.parameters()).device
    
    nlls = []
    loss_fn = CrossEntropyLoss(reduction='none')
    
    for i in tqdm(range(0, len(eval_dataset), batch_size), desc="Calculating perplexity"):
        # PTB dataset has sentences directly
        batch_texts = eval_dataset[i:i + batch_size]['sentence']
        
        # print(type(batch_texts), batch_texts.keys())
        
        # exit()
        
        encodings = tokenizer(batch_texts, 
                            max_length=max_length,
                            truncation=True,
                            padding=True,
                            return_tensors="pt")
        
        input_ids = encodings.input_ids.to(device)
        attention_mask = encodings.attention_mask.to(device)
        
        with torch.no_grad():
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits
            
            # Shift for next token prediction
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = input_ids[..., 1:].contiguous()
            
            # Calculate loss
            loss = loss_fn(shift_logits.view(-1, shift_logits.size(-1)), 
                        shift_labels.view(-1))
            
            # Apply mask to handle padding
            mask = attention_mask[..., 1:].contiguous().view(-1)
            loss = loss * mask
            
            # Average loss for each sequence
            seq_lengths = mask.sum(dim=-1)
            seq_loss = loss.view(input_ids.size(0), -1).sum(dim=-1) / seq_lengths
            nlls.extend(seq_loss.cpu().numpy())
            
    # Calculate perplexity
    ppl = np.exp(np.mean(nlls))
    return ppl


def evaluate_model_perplexity(model_name, quantize=False, layers_to_exclude=[]):
    """
    Evaluate model perplexity with optional quantization
    """
    # Load tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(model_name, 
                                                torch_dtype=torch.float32,
                                                trust_remote_code=True)
    
    # Load PTB dataset
    dataset = load_dataset("ptb_text_only", "penn_treebank", split="test", trust_remote_code=True)
    
    # Ensure dataset size is at least 3000 points
    if len(dataset) > 3000:
        dataset = dataset.select(range(3000))
    else:
        print(f"Warning: Dataset size ({len(dataset)}) is less than 3000 points")
    
    # Move model to GPU if available
    device = "cpu" # torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    
    t0 = time.time()
    generator = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
    )

    # Generate text
    prompt = "Write a short story about a robot learning to paint:"
    response = generator(
        prompt,
        max_length=100,
        num_return_sequences=1
    )

    # Print the generated text
    print('*'*80)
    print('response text: \n', response[0]['generated_text'])
    print('*'*80)
    tc1 = time.time()
    
    print("Model Stats before quantization:")
    previous_memory_footprint = model.get_memory_footprint()
    print("Footprint of the model in MBs: ", previous_memory_footprint/1e+6)
    print_model_info(model)
    print("Time taken to generate text before quantization: ", tc1-t0)
    
    
    # Calculate perplexity before quantization
    print("\nCalculating perplexity before quantization...", end='\r')
    ppl_before = calculate_perplexity(model, tokenizer, dataset)
    print(f"Perplexity before quantization: {ppl_before:.2f}   ")
    
    if quantize:
        print("\nQuantizing model...")
        always_exclude_layers = ["lm_head"]
        layers_to_exclude=always_exclude_layers + layers_to_exclude
        replace_linear_with_target_and_quantize(model, W8A16LinearLayer, layers_to_exclude)
        
        print("\nCalculating perplexity after quantization...")
        ppl_after = calculate_perplexity(model, tokenizer, dataset)
    else:
        ppl_after = None
    
    t0 = time.time()
    generator = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
    )

    # Generate text
    prompt = "Write a short story about a robot learning to paint:"
    response = generator(
        prompt,
        max_length=100,
        num_return_sequences=1
    )

    # Print the generated text
    print('*'*80)
    print('response text: \n', response[0]['generated_text'])
    print('*'*80)
    tc2 = time.time()
    print("Model Stats after quantization:")
    previous_memory_footprint = model.get_memory_footprint()
    print("Footprint of the model in MBs: ", previous_memory_footprint/1e+6)
    print_model_info(model)
    print("Time taken to generate text after quantization: ", tc2-t0)
    
    return ppl_before, ppl_after

def main():
    model_name = "microsoft/phi-3-mini-4k-instruct"
    
    # Run evaluation
    print(f"Evaluating model: {model_name}")
    
    layers_to_exclude = ["transformer.wte", "transformer.wpe", "transformer.ln_f"]
    
    if len(layers_to_exclude) > 0:
        print(f"Excluding layers: {layers_to_exclude}")
    print("=" * 50)
    
    # layers_to_exclude = []
    
    ppl_before, ppl_after = evaluate_model_perplexity(model_name, quantize=True, layers_to_exclude=layers_to_exclude)
    
    # Print results
    print("\nResults:")
    print("=" * 50)
    print(f"Perplexity before quantization: {ppl_before:.6f}")
    print(f"Perplexity after quantization: {ppl_after:.6f}")
    print(f"Perplexity difference: {ppl_after - ppl_before:.6f}")
    print(f"Relative perplexity change: {((ppl_after - ppl_before) / ppl_before) * 100:.6f}%")

if __name__ == "__main__":
    main()