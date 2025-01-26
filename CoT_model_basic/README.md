# CoT_model_basic
- Commenced: 6/1/25
- Concluded: 23/1/25

## Overview
The first project involved investigating into how a chain of thought (CoT) large language model (LLM) could be implemented. The programming language to be used is python as it is a well known language that has a plethora of libraries and api for training and using large language models along with the HuggingFace library. The sub-project consists of a large language CoT model pipeline utilising pytorch, along with a traning and inference process that has been optimised with parallel sampling. 

## Background
Firstly, to enhance my knowledge of LLMs and their implementation, I researched a number of videos and articles of LLM architecture, transformers and the specific parameters in modern large language models.
![Research into LLMs explicit notes](./images/LLM%20intro%20research.png "LLM research #1")
[Associated video by Andrej Karpathy](https://www.youtube.com/watch?v=zjkBMFhNj_g)

I was then tasked with researching deeper into chain of thought models, or what is more known in the industry as test-time scaling, where the accuracy and correctness of the inference of a model is dictated by the amount of time provided for the model to compute.
![Research into LLMs explicit notes](./images/Speculations%20on%20Test-Time%20Scaling.png "LLM research #2")
[Associated video by Sasha Rush](https://www.youtube.com/watch?v=6PEJ96k1kiw)

After this research, I generated some rough and rudimentry diagrams of potential architectures for a chain of thought model.
![Brainstorm sketch of potential CoT models](./images/CoT%20Architecture%20Brainstorm.png "Brainstorm of CoT models")

Upon convening with Tobin South, we came to the conclusion that a LLM that is trained to supply a thought process to its answers is the best approach to create a successful model. We discovered that a base model will be required to achieve this, prompting the next step of finding such models with CoT capabilities.
![Notes on CoT practicality](./images/CoT%20practicality.png "Practicality of a LLM that has CoT capabilities")

## Outcome
After all of the research, a basic implementation of a LLM in python was created. I decided to program a [localised program that can be run on any computer](./local/) to test the implementation as well as a [virtual machine variant using the vllm library](./vllm/) for suitable use on a server side GPU. Both programs were heavily influenced by [a LLM CoT implementation video by Trelis Research](https://www.youtube.com/watch?v=MvaUcc0mNOU), which also details the use of verifiers in training a CoT model. This was implemented expect for the training process, as this is rudimentary.

### Next steps
To find a base model with chain of thought architecture or capabilites that is able to be used open source such that I can create an inference endpoint and utilise the model in other programs or implementations through addtional training. This stage of development was initiated in [CoT_base_models](https://github.com/sam-beck/MIT-BigData-Living-Lab/tree/main/CoT_base_models).