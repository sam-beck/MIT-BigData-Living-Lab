# CoT_model_basic: vllm implementation
A LLM implementation that can be applied to CoT through multisampling and a verifier to dictate correct outputs on a virtual machine. The program utilises multithreading to increase the throughput of output judging and speed up training processes.

## [CoT_1](./CoT_1.py)
The verifier, multisampling and judging implementation.

## [model](./model.py)
A program that initalises the LLM using the vllm library.

## [apiKey](./apiKey.py)
A program that prompts the user to enter an OpenAI key to allow inference of an OpenAI verifier.
