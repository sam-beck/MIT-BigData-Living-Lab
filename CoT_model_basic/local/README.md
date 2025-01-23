# CoT_model_basic: local implementation
A LLM implementation that can be applied to CoT through multisampling and a verifier to dictate correct outputs on a local machine. The program utilises multithreading to increase the throughput of output judging and speed up training processes.

## [CoT_1](./CoT_1.py)
The verifier, multisampling and judging implementation.

## [model](./model.py)
A program containing two of the model classes, SamplingParameters (a class to contain all sampling parameters to be used) and LanguageModel (a class that allows for inference using a AutoModelForCausalLM model)

## [apiKey](./apiKey.py)
A program that prompts the user to enter an OpenAI key to allow inference of an OpenAI verifier.
