# import torch
# from transformers import AutoModelForCausalLM, GPT2Config

# def count_parameters_local_model(model_path):
#     try:
#         # Load the checkpoint
#         checkpoint = torch.load(model_path)
        
#         # Create a new model
#         config = GPT2Config.from_pretrained('gpt2')
#         model = AutoModelForCausalLM.from_pretrained('gpt2')
        
#         # Load just the model state dict from the checkpoint
#         model.load_state_dict(checkpoint['model_state_dict'])
        
#         # Count parameters
#         total_params = sum(p.numel() for p in model.parameters())
#         trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        
#         print(f"Total Parameters: {total_params:,}")
#         print(f"Trainable Parameters: {trainable_params:,}")
        
#         # Print additional info from checkpoint
#         print(f"\nAdditional Information:")
#         print(f"Epoch: {checkpoint['epoch']}")
#         print(f"Loss: {checkpoint['loss']}")
        
#         return model
        
#     except Exception as e:
#         print(f"Error loading model: {str(e)}")
#         return None

# # Usage
# model_path = "./prompt_tune_best_model.pt"
# model = count_parameters_local_model(model_path)

import torch
from transformers import AutoModelForCausalLM
from peft import PeftModel, PeftConfig

def count_lora_parameters(model_dir):
    try:
        # Load base model
        base_model = AutoModelForCausalLM.from_pretrained('gpt2')
        
        # Load LoRA model directly using PEFT
        lora_model = PeftModel.from_pretrained(
            base_model,
            model_dir,
            is_trainable=True
        )
        
        # Count parameters
        base_params = sum(p.numel() for p in base_model.parameters())
        lora_params = sum(p.numel() for p in lora_model.parameters() if p.requires_grad)
        total_params = sum(p.numel() for p in lora_model.parameters())
        
        print(f"\nParameter Counts:")
        print(f"Base Model Parameters: {base_params:,}")
        print(f"LoRA Trainable Parameters: {lora_params:,}")
        print(f"Total Parameters: {total_params:,}")
        print(f"Parameter Efficiency Ratio: {base_params/lora_params:.2f}x")
        
        # Print LoRA specific parameters
        print("\nLoRA Architecture:")
        for name, param in lora_model.named_parameters():
            if 'lora' in name.lower() and param.requires_grad:
                print(f"{name}: {param.numel():,} parameters")
        
        return lora_model
        
    except Exception as e:
        print(f"Error loading LoRA model: {str(e)}")
        return None

# Usage
path = "./LoRA_best_model.pt"  # This is actually a directory
model = count_lora_parameters(path)
