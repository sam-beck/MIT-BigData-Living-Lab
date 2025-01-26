# CoT_model_basic
- Commenced: 26/1/25
- Concluded: TBD

## Overview
Upon researching and learning about large language models, their implementation in python and chain of thought architecture, the next step was to find open-source CoT LLMs to then be used in an inference endpoint for ease of access. Furthermore, thought into training such models and datasets is also to be considered.

## Background
Research was done into creating and deploying an inference model (once found) in a web browser that was compatiable with the HuggingFace open library of models to broaden the number of models that are able to be used within the endpoint. After looking into [Microsoft Azure machine learning endpoint implementations](https://learn.microsoft.com/en-us/azure/machine-learning/concept-endpoints?view=azureml-api-2) and [HuggingFace's dedicated inference endpoint platform](https://huggingface.co/inference-endpoints/dedicated), I decided to attempt to implement [my own inference endpoint](./prototype_inference_endpoint/) utilising python's [Flask](https://flask.palletsprojects.com/en/stable/) library, along with the LLM inference I had implemented using pytorch.

## Outcome
The [prototype_inference_endpoint](./prototype_inference_endpoint/) sub-folder contains my first, proof-of-concept implementation of an inference endpoint, having options for use with multiple popular LLMs and the researched CoT models. 

### Next steps