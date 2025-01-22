# Parallel (multithreaded) sampling with a voting verifier for VMs
# Based on the works of Trelis Research: https://www.youtube.com/@TrelisResearch

import apiKey, model
import datasets
import os
from vllm import SamplingParams
from pydantic import BaseModel
import numpy as np
from concurrent.futures import ThreadPoolExecutor, as_completed

# Structured response format for the evaluation results
class ScoringResult(BaseModel):
    # Reason for the correctness evaluation and whether the prediction is correct
    reason: str
    correct: bool

def simple_eval_with_sampling(dataset, n_gen=1, temperature=0.7, top_p=1.0, min_p=0.0, top_k=-1, best_of=1, use_beam_search=False, max_tokens=512):
    """
    Perform evaluation with sampling on a dataset.
    
    Parameters:
    - dataset: The input dataset containing questions and answers.
    - n_gen: Number of generations per question.
    - temperature: Controls randomness of predictions.
    - top_p: Nucleus sampling cumulative probability cutoff.
    - min_p: Minimum cumulative probability cutoff.
    - top_k: Limits the sampling pool to the top-k tokens.
    - best_of: Number of best sequences to return.
    - use_beam_search: Whether to use beam search for generation.
    - max_tokens: Maximum length of generated sequences.

    Returns:
    - A dictionary with the total correct predictions and the percentage of correctness.
    """
    preds = []  # Store generated predictions
    answers = []  # Store ground-truth answers

    # Initialize sampling parameters
    samplingParameters = SamplingParams(
        n=n_gen,
        temperature=temperature,
        top_p=top_p,
        top_k=top_k,
        best_of=best_of,
        use_beam_search=use_beam_search,
        max_tokens=max_tokens
    )

    # Generate predictions for each question in the dataset
    for row in dataset:
        question = row["question"]
        answers.append(row["answer"])
        messages = [{"role": "user", "content": question}]
        outputs = model.llm.chat(messages, samplingParameters, use_tqdm=True)
        inferred = []

        for output in outputs[0].outputs:
            response = output.text
            if response is not None:
                inferred.append(response)
        preds.append(inferred)

    # Function to evaluate a single prediction against the true answer
    def judge_prediction(answer, prediction):
        numeric = answer.split("#### ")[-1]
        message = [
            {"role": "system", "content": "Evaluate the correctness of the predicted answer"},
            {"role": "user", "content": f"\nTrue answer: {answer}\nPredicted answer: {prediction}"}
        ]
        judge_result = apiKey.client.beta.chat.completions.parse(
            model="gpt-4o-mini",
            messages=message,
            response_format=ScoringResult,
            temperature=0,
            max_tokens=1024
        )
        return judge_result.choices[0].message.parsed

    correct_count = 0  # Counter for correct predictions
    num_workers = os.cpu_count() - 2  # Number of threads to use

    # Use a thread pool for parallel evaluation
    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        for index, answers in enumerate(preds):
            to_judge = {
                executor.submit(judge_prediction, answers[index], p): p
                for p in answers
            }

            correct = False
            for judged in as_completed(to_judge):
                try:
                    if judged.result():
                        correct_count += 1
                        break  # Stop checking once a correct prediction is found
                except Exception as exception:
                    print(f"Judging generated an exception: {exception}")

    total = len(dataset)
    percentage = correct_count / total * 100 if total > 0 else 0

    return {
        "Total correct": correct_count,
        "Percentage correct": percentage
    }

def run_simple_eval_with_multi_sampling(dataset, n_gen=16, m=5, temperature=0.7, top_p=1.0, min_p=0.0, top_k=-1, best_of=1, use_beam_search=False, max_tokens=512):
    """
    Perform multiple rounds of evaluation and calculate statistics.

    Parameters:
    - dataset: The input dataset for evaluation.
    - n_gen: Number of generations per question.
    - m: Number of evaluation rounds.
    - temperature: Controls randomness of predictions.
    - top_p: Nucleus sampling cumulative probability cutoff.
    - min_p: Minimum cumulative probability cutoff.
    - top_k: Limits the sampling pool to the top-k tokens.
    - best_of: Number of best sequences to return.
    - use_beam_search: Whether to use beam search for generation.
    - max_tokens: Maximum length of generated sequences.

    Returns:
    - A dictionary with mean and mean absolute deviation (MAD) for correct results and percentages.
    """
    correct = []
    percentage = []

    # Perform `m` rounds of evaluation
    for i in range(m):
        result = simple_eval_with_sampling(dataset, n_gen, temperature, top_p, min_p, top_k, best_of, use_beam_search, max_tokens)
        correct.append(result["Total correct"])
        percentage.append(result["Percentage correct"])

    # Calculate mean and MAD for correct results and percentages
    mean = np.mean(correct)
    mad = np.mean(np.abs(correct - mean))

    mean_percentage = np.mean(percentage)
    mad_percentage = np.mean(np.abs(percentage - mean_percentage))

    return {
        "mean for correct results": round(mean, 1),
        "mad for correct results": round(mad, 1),
        "mean for percentage results": round(mean_percentage, 1),
        "mad for percentage results": round(mad_percentage, 1),
    }

# Load the dataset
# Size of the dataset to use for evaluation
dataSize = 10

# Load a subset of the GSM8K dataset for evaluation and training
# Adjust the range as needed for training/testing split
dataset = datasets.load_dataset("openai/gsm8k", "main", split=f"train[0:{dataSize}]", trust_remote_code=True)
trainingDataSet = datasets.load_dataset("openai/gsm8k", "main", split=f"train[-{dataSize}:]", trust_remote_code=True)