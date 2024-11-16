import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer, logging
from datasets import load_dataset
from rouge_score import rouge_scorer
from tqdm import tqdm
import warnings

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
    
    model = GPT2LMHeadModel.from_pretrained('gpt2')
    checkpoint = torch.load(model_path, map_location=device)
    
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
    
    model = model.to(device)
    model.eval()  # Set to evaluation mode
    
    # Load dataset
    print("Loading dataset...")
    dataset = load_dataset('cnn_dailymail',
                        '3.0.0', cache_dir=scratch_dir)
    
    num_samples = int(len(dataset['test'])*0.1) if num_samples is None else num_samples
    
    print(num_samples)
        
    test_data = list(dataset['test'])[:num_samples]
    
    # Setup ROUGE scorer
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
    
    # Track scores
    all_scores = []
    
    total_loss = 0.0
    
    # Evaluate
    print("Generating summaries and calculating scores...")
    for item in tqdm(test_data, desc='Processing:', leave=False):
        try:
            # Get article and reference summary
            article = item['article'][:1024]  # Limit input length
            reference_summary = item['highlights']
            
            # Generate summary
            inputs = tokenizer(
                "Summarize: " + article, 
                return_tensors="pt",
                truncation=True,
                max_length=512
            ).to(device)
            
            with torch.no_grad():  # Disable gradient calculation
                outputs = model.generate(
                    inputs['input_ids'],
                    attention_mask=inputs['attention_mask'].to(device),
                    max_new_tokens=150,
                    num_beams=2,  # Reduced beam size
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
            
            # Print example
            # print(f"\nExample {len(all_scores)}:")
            # print(f"Generated: {generated_summary[:100]}...")
            # print(f"Reference: {reference_summary[:100]}...")
            # print(f"ROUGE scores: R1={scores['rouge1'].fmeasure:.3f}, "
            #     f"R2={scores['rouge2'].fmeasure:.3f}, "
            #     f"RL={scores['rougeL'].fmeasure:.3f}")
            
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
        
        print("\nAverage ROUGE scores:")
        print(f"ROUGE-1: {avg_scores['rouge1']:.3f}")
        print(f"ROUGE-2: {avg_scores['rouge2']:.3f}")
        print(f"ROUGE-L: {avg_scores['rougeL']:.3f}")
        print(f"Average Loss: {total_loss/len(all_scores):.3f}")
    else:
        print("No scores were calculated successfully.")

if __name__ == "__main__":
    model_path = './finetune_best_model.pt'  # Adjust this path
    evaluate_model(model_path)