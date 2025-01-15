model_name_map = {
    "gpt2": "GPT2-Small",
    "gpt2-medium": "GPT2-Med",
    "gpt2-large": "GPT2-Large",
    "gpt2-xl": "GPT2-XL",
    "Llama-2-7b-hf": "LLaMA-2-7b",
    "Llama-2-7b-hf": "LLaMA-2-7b",
    "Llama-2-7b-chat-hf": "LLaMA-2-7b-Instruct",
    "Llama-2-13b-hf": "LLaMA-2-13b",
    "Llama-2-13b-chat-hf": "LLaMA-2-13b-Instruct",
    "Meta-Llama-3.1-8B-Instruct": "LLaMA-3.1-8B-Instruct",
    "Llama-3.2-3B-Instruct": "LLaMA-3.2-3B-Instruct",
    "Phi-3.5-mini-instruct": "Phi-3.5-Mini-Instruct",
    "gemma-1.1-7b-it": "Gemma-1.1-7B-Instruct",
    "gemma-2b": "Gemma-2B",
    "gemma-7b": "Gemma-7B",
    "falcon-7b": "Falcon-7B",
    "falcon-7b-instruct": "Falcon-7B-Instruct",
    "Mistral-7B-Instruct-v0.3": "Mistral-7B-Instruct",
    "Mistral-7B-v0.3": "Mistral-7B",
}

def get_num_blocks(model_name):
    return {
        "gpt2": 12,
        "gpt2-medium": 24,
        "gpt2-large": 36,
        "gpt2-xl": 48,
        "Llama-2-7b-hf": 32,
        "Llama-2-7b-chat-hf": 32,
        "vicuna-7b-v1.3": 32,
        "Llama-2-13b-hf": 40,
        "Llama-2-13b-chat-hf": 40,
        "Meta-Llama-3.1-8B-Instruct": 32,
        "Llama-3.2-3B-Instruct": 28,

        "Phi-3.5-mini-instruct": 32,

        "falcon-7b": 32,
        "falcon-7b-instruct": 32,

        "Mistral-7B-v0.3": 32,
        "Mistral-7B-Instruct-v0.3": 32,

        "vicuna-13b-v1.3": 40,

        "gemma-2b": 18,
        "gemma-2b-it": 18,
        "gemma-2-2b": 26,

        "gemma-1.1-2b-it": 18,
        "gemma-7b": 28,
        "gemma-1.1-7b-it": 28,
    }[model_name]

def get_hidden_dim(model_name):
    return {
        "gpt2": 768,
        "gpt2-medium": 1024,
        "gpt2-large": 1280,
        "gpt2-xl": 1600,

        "Llama-2-7b-hf": 4096,
        "Llama-2-7b-chat-hf": 4096,

        "Llama-2-13b-hf": 5120,
        "Llama-2-13b-chat-hf": 5120,

        "Meta-Llama-3.1-8B-Instruct": 4096,

        "Llama-3.2-3B-Instruct": 3072,

        "Phi-3.5-mini-instruct": 3072,

        "gemma-1.1-7b-it": 3072,
        
        "gemma-2-2b": 2304,
        "gemma-2-9b": 3072,

        "gemma-2b": 2048,
        "gemma-2b-it": 2048,
        "gemma-7b": 3072,

        "falcon-7b": 4544,
        "falcon-7b-instruct": 4544,

        "Mistral-7B-v0.3": 4096,
        "Mistral-7B-Instruct-v0.3": 4096,
    }[model_name]


def get_layer_names(model_name):
    num_blocks = get_num_blocks(model_name)

    if "gpt2" in model_name or "falcon" in model_name:
        return [f'transformer.h.{block}' 
            for block in range(num_blocks) 
            # for layer_desc in ['ln_1', 'attn', 'ln_2', 'mlp']
        ]
    elif "Llama" in model_name or "gemma" in model_name or "Phi" in model_name or "Mistral" in model_name:
        return [f'model.layers.{layer_num}' 
            for layer_num in range(num_blocks) 
            # for layer_desc in ["input_layernorm", "self_attn", "post_attention_layernorm", "mlp"]
        ]  
    else:
        raise ValueError(f"{model_name} not supported currently!")
