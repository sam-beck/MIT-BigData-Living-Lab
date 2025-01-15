# Creates the LLM to be trained, along with querying its dataset
import os
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

class SamplingParameters:
    def __init__(self,max_length=50,min_length=0,temperature=1.0,top_k=50,top_p=1.0,num_return_sequences=1,repetition_penalty=1.0,no_repeat_ngram_size=0,do_sample=True,early_stopping=False,num_beams=1,best_of=1,pad_token_id=None,eos_token_id=None):
        self.max_length = max_length
        self.min_length = min_length
        self.temperature = temperature
        self.top_k = top_k
        self.top_p = top_p
        self.num_return_sequences = num_return_sequences
        self.repetition_penalty = repetition_penalty
        self.no_repeat_ngram_size = no_repeat_ngram_size
        self.do_sample = do_sample
        self.early_stopping = early_stopping
        self.num_beams = num_beams
        self.best_of = best_of
        self.pad_token_id = pad_token_id
        self.eos_token_id = eos_token_id
        
class LanguageModel:
    def __init__(self, model_name="gpt2", device=None):
        self.model_name = model_name
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.model = AutoModelForCausalLM.from_pretrained(self.model_name)
        self.model.to(self.device)
    
    def chat(self, prompt, samplingParams : SamplingParameters, custom_attention_mask=None):
        if isinstance(prompt, dict) and "role" in prompt and "content" in prompt:
            if prompt["role"] != "user":
                raise ValueError(f"Unsupported role prompt: {prompt['role']} used.")
            prompt = prompt["content"]
        
        returnSequences = max(samplingParams.num_return_sequences, samplingParams.best_of)

        inputs = self.tokenizer(prompt,return_tensors="pt",padding=True,truncation=True,max_length=samplingParams.max_length).to(self.device)
        attention_mask = custom_attention_mask if custom_attention_mask is not None else inputs.attention_mask

        outputs = self.model.generate(
            inputs.input_ids,
            attention_mask=attention_mask,
            max_length=samplingParams.max_length,
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

         # Decode outputs and calculate scores
        decoded_outputs = [self.tokenizer.decode(output, skip_special_tokens=True) for output in outputs]
        scores = outputs.scores if hasattr(outputs, 'scores') else None
        
        # Select top `best_of` sequences based on scores if available
        if scores is not None and samplingParams.best_of > 1:
            ratedSequences = zip(decoded_outputs, scores)
            ratedSequences = sorted(ratedSequences, key=lambda x: -x[1])  # Sort by score descending
            bestSequences = [seq for seq, _ in ratedSequences[:samplingParams.best_of]]
        else:
            bestSequences = decoded_outputs[:samplingParams.best_of]
        
        return bestSequences

# Enables transfer from hugging face
os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"

# Load model and tokenizer
model_name = "gpt2"  # Replace with the desired model
llm = LanguageModel(model_name=model_name)

#TODO delete below, for test purposes only

params = SamplingParameters( max_length=64,
        num_return_sequences=5,
        best_of=3,  # Return the top 3 sequences
        temperature=0.7,
        top_p=0.9,
        repetition_penalty=1.2,
        no_repeat_ngram_size=2,
        do_sample=True  # Sampling mode must be enabled for diverse outputs
        )

results = llm.chat({"role": "user", "content": "What is the capital of France?"}, params)

for i, result in enumerate(results):
        print(f"Best Text {i + 1}:\n{result}\n")