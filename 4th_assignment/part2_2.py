import gc
import time
import json
import torch
import psutil
import numpy as np
import transformers
import bitsandbytes as bnb
from tqdm.auto import tqdm
from torch.nn import CrossEntropyLoss
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

def check_cuda_availability():
    """Check CUDA availability and initialize it"""
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available. Please check your GPU installation.")
    
    # Initialize CUDA
    torch.cuda.init()
    device = torch.device("cuda")
    
    # Print CUDA information
    print(f"Using GPU: {torch.cuda.get_device_name(0)}")
    print(f"CUDA Version: {torch.version.cuda}")
    return device

def get_memory_usage():
    """Get current memory usage in GB"""
    process = psutil.Process()
    cpu_memory = process.memory_info().rss / (1024 * 1024 * 1024)
    
    if torch.cuda.is_available():
        gpu_memory = torch.cuda.memory_allocated() / (1024 * 1024 * 1024)
        return cpu_memory, gpu_memory
    return cpu_memory, 0

def measure_inference_latency(model, tokenizer, input_text, num_runs=10):
    """Measure average inference latency"""
    try:
        inputs = tokenizer(input_text, return_tensors="pt")
        inputs = {k: v.to(model.device) for k, v in inputs.items()}
        
        # Measure latency
        latencies = []
        for _ in tqdm(range(num_runs), desc="Measuring Latency", leave=False):
            torch.cuda.synchronize()  # Ensure CUDA operations are complete
            start_time = time.time()
            with torch.no_grad():
                _ = model.generate(**inputs, max_new_tokens=20)
            torch.cuda.synchronize()  # Ensure CUDA operations are complete
            latencies.append(time.time() - start_time)
        
        return np.mean(latencies)
    except Exception as e:
        print(f"Error during inference: {str(e)}")
        raise

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


def load_nf4_model(model_name):
    nf4_config = transformers.BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
    )

    model_nf4 = AutoModelForCausalLM.from_pretrained(
                model_name,
                device_map="auto",
                torch_dtype=torch.float16,
                quantization_config=nf4_config
            )
    
    return model_nf4

def main():
    try:
        # Load model and tokenizer
        model_name = "microsoft/phi-3-mini-4k-instruct"
        print(f"Loading {model_name}...")
        
        # Load tokenizer
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        
        # Load dataset
        print("Loading Wikipedia dataset...")
        
        # Test input
        test_input = "The quick brown fox jumps over the lazy dog"
        # Load PTB dataset
        dataset = load_dataset("ptb_text_only", "penn_treebank", split="test", trust_remote_code=True)
        
        # Ensure dataset size is at least 3000 points
        if len(dataset) > 3000:
            dataset = dataset.select(range(3000))
        else:
            print(f"Warning: Dataset size ({len(dataset)}) is less than 3000 points")
        
        # Dictionary to store results
        results = {}
        
        # Load appropriate model
        model = load_nf4_model(model_name)
        model_memory_footprint = model.get_memory_footprint()
        
        # Measure metrics
        cpu_mem, gpu_mem = get_memory_usage()
        print('memory uasge computed.')
        latency = measure_inference_latency(model, tokenizer, test_input)
        print(f"\tCalculating perplexity for nf4 model...", end='\r')
        perplexity = calculate_perplexity(model, tokenizer, dataset)
        print(f"\tCalculated perplexity for nf4 model               ")
        
        # Store results
        results['nf4'] = {
                "cpu_memory": float(cpu_mem),
                "gpu_memory": float(gpu_mem),
                "latency": float(latency),
                "perplexity": float(perplexity),
                "model_memory_footprint": float(model_memory_footprint)
            }
        
        # Clean up
        del model
        gc.collect()
        torch.cuda.empty_cache()
            
        # Print results
        print("\n=== Results ===")
        for model_type, metrics in results.items():
            print(f"\n{model_type.upper()} MODEL:")
            print(f"\tCPU Memory: {metrics['cpu_memory']:.2f} GB")
            print(f"\tGPU Memory: {metrics['gpu_memory']:.2f} GB")
            print(f"\tInference Latency: {metrics['latency']:.4f} seconds")
            print(f"\tPerplexity: {metrics['perplexity']:.2f}")
            
            print("="*80)
        # dump results to json file
        with open('2_2_output.json', 'w') as json_file:
            json.dump(results, json_file)
            
    except Exception as e:
        print(f"An error occurred: {str(e)}")
        raise
    

if __name__ == "__main__":
    main()
