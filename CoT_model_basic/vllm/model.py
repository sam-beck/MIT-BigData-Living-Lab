# Creates the LLM to be trained, along with querying its dataset
import os
from vllm import LLM, SamplingParams

# Enables transfer from hugging face
os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"

# Sampling params object, temperature = 0 so quite strict
samplingParameters = SamplingParams(temperature=0)

# Get pre-trained LLM
llm = LLM(model = "microsoft/Phi-3-mini-128k-instruct", dtype="half", max_model_len=4096)