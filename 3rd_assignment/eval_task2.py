import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer, logging
from datasets import load_dataset
from rouge_score import rouge_scorer
from tqdm import tqdm
from peft import PeftModel, PeftConfig
import warnings
import os

# Ignore all warnings
warnings.filterwarnings("ignore")
logging.set_verbosity_error()

scratch_dir = "/scratch/hmnshpl/anlp_data"

def evaluate_model(model_path, num_samples=None):
    # Force CPU usage
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Load model and tokenizer
    print("Loading model and tokenizer...")
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    tokenizer.pad_token = tokenizer.eos_token
    
    # Load the base model
    base_model = GPT2LMHeadModel.from_pretrained('gpt2')
    
    # Load the LoRA adapter config and model
    print(f"Loading LoRA adapter from {model_path}...")
    config = PeftConfig.from_pretrained(model_path)
    model = PeftModel.from_pretrained(base_model, model_path)
    
    model = model.to(device)
    model.eval()  # Set to evaluation mode
    
    # Load dataset
    print("Loading dataset...")
    dataset = load_dataset('cnn_dailymail', '3.0.0', cache_dir=scratch_dir)
    
    num_samples = int(len(dataset['test'])*0.1) if num_samples is None else num_samples
    print(f"Evaluating on {num_samples} samples")
    
    test_data = list(dataset['test'])[:num_samples]
    
    # Setup ROUGE scorer
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
    
    # Track scores and loss
    all_scores = []
    total_loss = 0.0
    num_loss_samples = 0
    
    # Evaluate
    print("Generating summaries and calculating scores...")
    for item in tqdm(test_data, desc='Processing:', leave=False):
        try:
            # Get article and reference summary
            article = item['article'][:1024]  # Limit input length
            reference_summary = item['highlights']
            
            # Prepare input for loss calculation
            full_text = f"Summarize: {article} Summary: {reference_summary}"
            inputs = tokenizer(
                full_text,
                return_tensors="pt",
                truncation=True,
                max_length=512
            ).to(device)
            
            # Calculate loss
            with torch.no_grad():
                outputs = model(
                    input_ids=inputs['input_ids'],
                    attention_mask=inputs['attention_mask'],
                    labels=inputs['input_ids']
                )
                loss = outputs.loss
                total_loss += loss.item()
                num_loss_samples += 1
            
            # Generate summary
            gen_inputs = tokenizer(
                "Summarize: " + article, 
                return_tensors="pt",
                truncation=True,
                max_length=512
            ).to(device)
            
            with torch.no_grad():
                outputs = model.generate(
                    gen_inputs['input_ids'],
                    attention_mask=gen_inputs['attention_mask'],
                    max_new_tokens=150,
                    num_beams=2,
                    length_penalty=1.5,
                    early_stopping=True,
                    pad_token_id=tokenizer.eos_token_id,
                    no_repeat_ngram_size=2
                )
            
            generated_summary = tokenizer.decode(outputs[0], skip_special_tokens=True)
            if generated_summary.startswith("Summarize: "):
                generated_summary = generated_summary[len("Summarize: "):]
            
            # Calculate ROUGE scores
            scores = scorer.score(reference_summary, generated_summary)
            
            all_scores.append({
                'rouge1': scores['rouge1'].fmeasure,
                'rouge2': scores['rouge2'].fmeasure,
                'rougeL': scores['rougeL'].fmeasure
            })
            
            # Print example (uncomment for debugging)
            # print(f"\nExample {len(all_scores)}:")
            # print(f"Generated: {generated_summary[:100]}...")
            # print(f"Reference: {reference_summary[:100]}...")
            
            # Clear memory
            torch.cuda.empty_cache() if torch.cuda.is_available() else None
            
        except Exception as e:
            print(f"Error processing example: {str(e)}")
            continue

    # Calculate and print average scores
    if all_scores:
        avg_scores = {
            metric: sum(score[metric] for score in all_scores) / len(all_scores)
            for metric in ['rouge1', 'rouge2', 'rougeL']
        }
        
        avg_loss = total_loss / num_loss_samples if num_loss_samples > 0 else float('inf')
        
        print("\nEvaluation Results:")
        print(f"Test Loss: {avg_loss:.4f}")
        print("\nROUGE Scores:")
        print(f"ROUGE-1: {avg_scores['rouge1']:.4f}")
        print(f"ROUGE-2: {avg_scores['rouge2']:.4f}")
        print(f"ROUGE-L: {avg_scores['rougeL']:.4f}")
        
        # Save results to file
        results_file = os.path.join(os.path.dirname(model_path), "evaluation_results.txt")
        with open(results_file, "w") as f:
            f.write("Evaluation Results:\n")
            f.write(f"Test Loss: {avg_loss:.4f}\n")
            f.write("\nROUGE Scores:\n")
            f.write(f"ROUGE-1: {avg_scores['rouge1']:.4f}\n")
            f.write(f"ROUGE-2: {avg_scores['rouge2']:.4f}\n")
            f.write(f"ROUGE-L: {avg_scores['rougeL']:.4f}\n")
    else:
        print("No scores were calculated successfully.")

if __name__ == "__main__":
    model_path = './best_model.pt'
    evaluate_model(model_path)