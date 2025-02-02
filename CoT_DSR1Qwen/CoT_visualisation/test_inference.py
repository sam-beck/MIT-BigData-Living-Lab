import model
model_name = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"
llm = model.LanguageModel(model_name=model_name)

samplingParameters = model.SamplingParameters(top_p=0.2,temperature=0.1,top_k=2,best_of=1,num_beams=1,max_length=150)
results = llm.generate("If a train travels at 60 mph for 2.5 hours, how far does it travel?",samplingParameters)
for text in results:
    print("\n"+text+"\n")