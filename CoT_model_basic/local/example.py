import model

params = model.SamplingParameters( 
    max_length=64,
    num_return_sequences=5,
    best_of=3,  # Return the top 3 sequences
    temperature=0.3,
    top_p=0.7,
    top_k=2,
    repetition_penalty=1.2,
    no_repeat_ngram_size=2,
    do_sample=True  # Sampling mode must be enabled for diverse outputs
    )

results = model.llm.chat({"role": "user", "content": "The capital of Australia is"}, params)

for i, result in enumerate(results):
        print(f"Best Text {i + 1}:\n{result}\n")