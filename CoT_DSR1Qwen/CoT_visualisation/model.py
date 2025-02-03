# Creates the LLM to be trained, along with querying its dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

# Sampling parameters class for LanguageModel
class SamplingParameters:
    def __init__(self, max_new_tokens=50, max_length=512, min_length=0, temperature=0.9, top_k=50, top_p=1.0, num_return_sequences=1, repetition_penalty=1.0, no_repeat_ngram_size=0, do_sample=True, early_stopping=False, num_beams=1, best_of=1, pad_token_id=None, eos_token_id=None):
        # The maximum length of the tokenized sequence.    
        self.max_length = max_length

        # The maximum length of the generated sequence.
        self.max_new_tokens = max_new_tokens
        
        # The minimum length of the generated sequence.
        self.min_length = min_length
        
        # Controls the randomness of predictions by scaling the logits before applying softmax.
        self.temperature = temperature
        
        # Limits the sampling pool to the top-k most probable tokens.
        self.top_k = top_k
        
        # Filters the sampling pool to include only tokens whose cumulative probability is <= top_p.
        self.top_p = top_p
        
        # The number of output sequences to generate.
        self.num_return_sequences = num_return_sequences
        
        # Penalizes repeated tokens to reduce repetitiveness in the output.
        self.repetition_penalty = repetition_penalty
        
        # Ensures no n-gram of this size is repeated in the generated sequence.
        self.no_repeat_ngram_size = no_repeat_ngram_size
        
        # If True, sampling is used for text generation; if False, greedy decoding is used.
        self.do_sample = do_sample
        
        # If True, stops generation once an EOS token is generated for all beams.
        self.early_stopping = early_stopping
        
        # The number of beams for beam search; higher values improve quality but increase computation.
        self.num_beams = num_beams
        
        # The number of best sequences returned when using beam search.
        self.best_of = best_of
        
        # The token ID used for padding shorter sequences.
        self.pad_token_id = pad_token_id
        
        # The token ID representing the end of a sequence.
        self.eos_token_id = eos_token_id
        
class LanguageModel:
    def __init__(self, model_name="gpt2", device=None):
        # The name of the pretrained model to be used (e.g., GPT-2).
        self.model_name = model_name
        
        # The device to run the model on (e.g., "cuda" for GPU or "cpu").
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        
        # Load the tokenizer associated with the model.
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        
        # Set the padding token to the end-of-sequence token if not already set.
        self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Load the pretrained language model.
        self.model = AutoModelForCausalLM.from_pretrained(self.model_name, torch_dtype=torch.float16, device_map="auto")
        
    def generate(self, prompt, samplingParams: SamplingParameters, custom_attention_mask=None):
        # Check if the prompt contains a role and content and validate the role.
        if ('role' in prompt or "role" in prompt) and ('content' in prompt or "content" in prompt):
            if prompt["role"] != "user":
                raise ValueError(f"Unsupported role prompt: {prompt['role']} used.")
            # Use only the content part of the prompt.
            prompt = prompt["content"]
        
        # Determine the number of sequences to return (at least `best_of` or `num_return_sequences`).
        returnSequences = max(samplingParams.num_return_sequences, samplingParams.best_of)

        # Tokenize the input prompt, ensuring proper padding and truncation.
        inputs = self.tokenizer(prompt, return_tensors="pt", padding=True, truncation=True, max_length=samplingParams.max_length).to(self.device)
        
        # Use a custom attention mask if provided; otherwise, use the default attention mask.
        attention_mask = custom_attention_mask if custom_attention_mask is not None else inputs.attention_mask

        # Generate responses using the model with the provided sampling parameters.
        outputs = self.model.generate(
            inputs.input_ids,
            attention_mask=attention_mask,
            max_new_tokens=samplingParams.max_new_tokens,
            min_length=samplingParams.min_length,
            temperature=samplingParams.temperature,
            top_k=samplingParams.top_k,
            top_p=samplingParams.top_p,
            num_return_sequences=returnSequences,
            repetition_penalty=samplingParams.repetition_penalty,
            no_repeat_ngram_size=samplingParams.no_repeat_ngram_size,
            do_sample=samplingParams.do_sample,
            early_stopping=samplingParams.early_stopping,
            num_beams=samplingParams.num_beams,
            pad_token_id=samplingParams.pad_token_id or self.tokenizer.eos_token_id,
            eos_token_id=samplingParams.eos_token_id or self.tokenizer.eos_token_id,
        )

        # Decode the generated outputs into human-readable text.
        decoded_outputs = [self.tokenizer.decode(output, skip_special_tokens=True) for output in outputs]

        # Extract scores from the outputs if available.
        scores = outputs.scores if hasattr(outputs, 'scores') else None
        
        # Select the top `best_of` sequences based on scores, if scores are provided.
        if scores is not None and samplingParams.best_of > 1:
            ratedSequences = zip(decoded_outputs, scores)
            ratedSequences = sorted(ratedSequences, key=lambda x: -x[1])  # Sort by score descending
            bestSequences = [seq for seq, _ in ratedSequences[:samplingParams.best_of]]
        else:
            bestSequences = decoded_outputs[:samplingParams.best_of]
        
        # Return the selected best sequences.
        return bestSequences