# CoT_DSR1Qwen
- Commenced: 2/2/25
- Concluded: 10/2/25

## Overview
After creating an inference endpoint, further developments were made into multi-sampling CoT models and creating a "tree" of thought paths. Using a larger llm and generating multiple samples by using a temperature parameter below 1.0, variations in the output thought processes can be achieved. A lightweight pre-trained CoT reasoning model was used for testing of the algorithm (DSR1-Distill-Qwen-1.5B).

## Outcome
The [CoT_DSR1Qwen](./CoTDSR1Qwen/) sub-folder contains the first CoT tree sampling technique, which utilises temperature to generate a number of outputs from the model and create a tree-like thought pattern.

### Next steps
The process is of exponential order, which is most definitely not ideal. However, more development is required into abandoning thought paths using verifiers along with the sequence confidence levels / scores. Such discarded thought paths can then be used for reinforcement learning (RL) of the model, which is the next step in fine-tuning a CoT model to implement such an llm in specific applications.