# Creates the LLM to be trained, along with querying its dataset
import os, datasets
from vllm import LLM, SamplingParams

# Enables transfer from hugging face
os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"

# Size of the dataset
dataSize = 10
dataset = datasets.load_dataset("openai/gsm8k","main",split=f"train[0:{dataSize}]",trust_remote_code=True)
trainingDataSet = datasets.load_dataset("openai/gsm8k","main",split=f"train[-{dataSize}:]",trust_remote_code=True)

# Sampling params object, temperature = 0 so quite strict
samplingParameters = SamplingParams(temperature=0)

# Get pre-trained LLM
llm = LLM(model = "microsoft/Phi-3-mini-128k-instruct", dtype="half", max_model_len=4096)