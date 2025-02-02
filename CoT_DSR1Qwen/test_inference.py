import torch
import matplotlib.pyplot as plt
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer

# Load model and tokenizer
MODEL_NAME = "deepseek-ai/deepseek-r1-distill-qwen-1.5b"
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME, torch_dtype=torch.float16, device_map="auto", output_attentions=True
)

def extract_chain_of_thought_with_attention(prompt, max_new_tokens=100):
    """
    Performs inference while extracting the step-by-step reasoning (chain of thought)
    and logs attention weights for each generated token.

    Args:
        prompt (str): Input question or statement prompting CoT reasoning.
        max_new_tokens (int): Maximum number of new tokens to generate.

    Returns:
        List[str]: Extracted chain of thought steps.
        List[torch.Tensor]: Attention weights for each generated token.
    """
    cot_prompt = f"{prompt}\nLet's think step by step.\n"

    # Tokenize input
    inputs = tokenizer(cot_prompt, return_tensors="pt").to(model.device)

    # Perform inference with attention tracking
    outputs = model.generate(
        **inputs, max_new_tokens=max_new_tokens, do_sample=False, output_attentions=True, return_dict_in_generate=True
    )

    # Decode generated text
    output_text = tokenizer.decode(outputs.sequences[0], skip_special_tokens=True)

    # Extract CoT steps
    cot_steps = []
    for line in output_text.split("\n"):
        if line.strip() and not line.startswith(prompt):  # Ignore input prompt
            cot_steps.append(line.strip())

    # Extract attention weights
    attentions = outputs.attentions  # List of attention matrices per layer

    return cot_steps, attentions

def visualize_multiple_attention_heads(attentions, tokenized_input, layers=[0, -1], heads=[0, 1, 2, 3]):
    """
    Visualizes attention across multiple layers and heads.
    
    Args:
        attentions (List[torch.Tensor]): Attention weights from inference.
        tokenized_input (List[str]): Tokenized input words.
        layers (List[int]): List of layer indices to compare.
        heads (List[int]): List of head indices to compare.
    """
    num_layers = len(layers)
    num_heads = len(heads)

    fig, axes = plt.subplots(num_layers, num_heads, figsize=(num_heads * 4, num_layers * 4))

    for i, layer in enumerate(layers):
        for j, head in enumerate(heads):
            ax = axes[i, j] if num_layers > 1 else axes[j]  # Handle 1-row case
            attention_matrix = attentions[layer][0][head].cpu().detach().numpy()

            ax.imshow(attention_matrix, cmap="viridis", interpolation="nearest")
            token_labels = tokenized_input[:attention_matrix.shape[0]]

            ax.set_xticks(range(len(token_labels)))
            ax.set_yticks(range(len(token_labels)))
            ax.set_xticklabels(token_labels, rotation=90, fontsize=8)
            ax.set_yticklabels(token_labels, fontsize=8)
            ax.set_title(f"Layer {layer}, Head {head}")

    plt.tight_layout()
    plt.show()

# Example usage
if __name__ == "__main__":
    prompt = "If a train travels at 60 mph for 2.5 hours, how far does it go?"
    cot_steps, attentions = extract_chain_of_thought_with_attention(prompt)

    print("\nChain of Thought Reasoning:")
    for i, step in enumerate(cot_steps, 1):
        print(f"Step {i}: {step}")

    # Tokenize for attention visualization
    tokenized_input = tokenizer.convert_ids_to_tokens(tokenizer(prompt)["input_ids"])
    
    # Visualize attention for multiple layers and heads
    visualize_multiple_attention_heads(attentions, tokenized_input, layers=[0, -1], heads=[0, 1, 2, 3])