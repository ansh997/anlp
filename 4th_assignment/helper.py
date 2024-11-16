import torch
# from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers import GPT2LMHeadModel, GPT2Tokenizer
from datasets import load_dataset
import numpy as np
import torch.nn.functional as F
import torch.nn as nn

def get_model_size(model):
    model_size = sum(p.numel() for p in model.parameters()) * model.element_size()
    return model_size / (1024 ** 2)

def get_model_tokenizer():
    model = GPT2LMHeadModel.from_pretrained("gpt2")
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    return model, tokenizer

def get_data(tokenizer):
    dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="train[:3000]")
    dataset = dataset.map(lambda x: tokenizer(x["text"],
                        return_tensors="pt"), batched=True)
    return dataset

def get_q_scale_and_zero_point(tensor, dtype=torch.int8):
    
    q_min, q_max = torch.iinfo(dtype).min, torch.iinfo(dtype).max
    r_min, r_max = tensor.min().item(), tensor.max().item()

    scale = (r_max - r_min) / (q_max - q_min)

    zero_point = q_min - (r_min / scale)

    # clip the zero_point to fall in [quantized_min, quantized_max]
    if zero_point < q_min:
        zero_point = q_min
    elif zero_point > q_max:
        zero_point = q_max
    else:
        # round and cast to int
        zero_point = int(round(zero_point))
    
    return scale, zero_point

def linear_q_with_scale_and_zero_point(
    tensor, scale, zero_point, dtype = torch.int8):

    scaled_and_shifted_tensor = tensor / scale + zero_point

    rounded_tensor = torch.round(scaled_and_shifted_tensor)

    q_min = torch.iinfo(dtype).min
    q_max = torch.iinfo(dtype).max

    q_tensor = rounded_tensor.clamp(q_min,q_max).to(dtype)
    
    return q_tensor

def w8_a16_forward(weight, input, scales, bias=None):
    
    casted_weights = weight.to(input.dtype)
    output = F.linear(input, casted_weights) * scales
    
    if bias is not None:
        output = output + bias
    return output

class W8A16LinearLayer(nn.Module):
    def __init__(self, in_features, out_features, 
                bias=True, dtype=torch.float32):
        super().__init__()
        
        
        self.register_buffer(
            "int8_weights",
            torch.randint(
                -128, 127, (out_features, in_features), dtype=torch.int8
            )
        )
        
        self.register_buffer("scales", 
                            torch.randn((out_features), dtype=dtype))
        
        if bias:
            self.register_buffer("bias", 
                                torch.randn((1, out_features), 
                                            dtype=dtype))
    
        else:
            self.bias = None

    def quantize(self, weights):
        w_fp32 = weights.clone().to(torch.float32)

        scales = w_fp32.abs().max(dim=-1).values / 127
        scales = scales.to(weights.dtype)

        int8_weights = torch.round(weights
                        /scales.unsqueeze(1)).to(torch.int8)

        self.int8_weights = int8_weights
        self.scales = scales
    
    def forward(self, input):
        return w8_a16_forward(self.int8_weights, 
                            input, self.scales, self.bias)   

def replace_linear_with_target_and_quantize(module, 
                            target_class, module_name_to_exclude):
    # Scales remain in float32, Biases remain in float32, weights are in int8, Activations are in float16
    for name, child in module.named_children():
        if isinstance(child, nn.Linear) and not \
        any([x == name for x in module_name_to_exclude]):
            old_bias = child.bias
            old_weight = child.weight

            new_module = target_class(child.in_features, 
                                    child.out_features, 
                                    old_bias is not None, 
                                    child.weight.dtype)
            setattr(module, name, new_module)

            getattr(module, name).quantize(old_weight)
            
            if old_bias is not None:
                getattr(module, name).bias = old_bias
        else:
            # Recursively call the function for nested modules
            replace_linear_with_target_and_quantize(child, 
                    target_class, module_name_to_exclude)

def get_detailed_model_size(model):
    """
    Get detailed memory usage by dtype, including buffers
    """
    def dtype_size_bytes(dtype):
        if dtype == torch.int8:
            return 1
        elif dtype in [torch.float16, torch.bfloat16]:
            return 2
        elif dtype == torch.float32:
            return 4
        else:
            return 8  # default for other types

    size_dict = {
        'float32': 0,
        'float16': 0,
        'int8': 0,
        'other': 0
    }
    
    # Check parameters
    for name, param in model.named_parameters():
        dtype = param.dtype
        size_bytes = param.nelement() * dtype_size_bytes(dtype)
        
        if dtype == torch.float32:
            size_dict['float32'] += size_bytes
        elif dtype == torch.float16:
            size_dict['float16'] += size_bytes
        elif dtype == torch.int8:
            size_dict['int8'] += size_bytes
        else:
            size_dict['other'] += size_bytes
            
    # Check buffers (important for quantized models)
    for name, buffer in model.named_buffers():
        dtype = buffer.dtype
        size_bytes = buffer.nelement() * dtype_size_bytes(dtype)
        
        if dtype == torch.float32:
            size_dict['float32'] += size_bytes
        elif dtype == torch.float16:
            size_dict['float16'] += size_bytes
        elif dtype == torch.int8:
            size_dict['int8'] += size_bytes
        else:
            size_dict['other'] += size_bytes
    
    # Convert to MB
    for key in size_dict:
        size_dict[key] = size_dict[key] / (1024 * 1024)
        
    return size_dict

def print_model_info(model):
    """
    Print detailed model information including quantization
    """
    print("\nDetailed Model Analysis:")
    print("-" * 50)
    
    # Count layer types
    layer_types = {}
    for _, module in model.named_modules():
        module_type = type(module).__name__
        if module_type not in layer_types:
            layer_types[module_type] = 0
        layer_types[module_type] += 1
    
    print("Layer Types:")
    for layer_type, count in layer_types.items():
        print(f"- {layer_type}: {count}")
    
    # Get memory usage by dtype
    memory_usage = get_detailed_model_size(model)
    print("\nMemory Usage by dtype (MB):")
    for dtype, size in memory_usage.items():
        print(f"- {dtype}: {size:.2f} MB")
    
    # Check specific W8A16LinearLayer properties
    w8a16_layers = [m for m in model.modules() if isinstance(m, W8A16LinearLayer)]
    if w8a16_layers:
        sample_layer = w8a16_layers[0]
        print("\nW8A16LinearLayer Properties:")
        print(f"- int8_weights dtype: {sample_layer.int8_weights.dtype}")
        print(f"- scales dtype: {sample_layer.scales.dtype}")
        if sample_layer.bias is not None:
            print(f"- bias dtype: {sample_layer.bias.dtype}")


if __name__ == "__main__":
    print("Helper functions for quantization")