# Parallel (multithreaded) sampling with a voting verifier
# Based on the works of Trelis Research: https://www.youtube.com/@TrelisResearch
import apiKey, model
import os, datasets
from pydantic import BaseModel
import numpy as np
from concurrent.futures import ThreadPoolExecutor, as_completed

# Enables transfer from hugging face
os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"

# Load model and tokenizer
model_name = "gpt2"  # Replace with the desired model
llm = model.LanguageModel(model_name=model_name)

# Structured response format
class ScoringResult(BaseModel):
        reason: str
        correct: bool

# Evaluates a prompt with sampling given n_gen, the number of return sequences. 
# This is then judged using gpt-4o-mini and if one correct answer is found, the result is correct. 
def simple_eval_with_sampling(dataset, n_gen=1, temperature=0.25, top_p=1.0, top_k=2, best_of=1, num_beams=1,max_tokens=512):
        # An array for each generated prediction output subarray
        preds = []
        # An array for the actual answers
        answers = []

        # Generate sampling parameters given params to function
        samplingParameters = model.SamplingParameters(num_return_sequences=n_gen,temperature=temperature,top_p=top_p,top_k=top_k,best_of=best_of,num_beams=num_beams,max_length=max_tokens,do_sample=False if num_beams > 1 else True)
        for row in dataset:
                # Question of given dataset prompt
                question = row["question"]
                # Add answer to prompt to answers array
                answers.append(row["answer"])
                # Get predicted outputs by inferring
                outputs = llm.chat({"role": "user", "content": question},samplingParameters)
                inferred = []
                for output in outputs:
                        # Check if outputs are null
                        if output is not None:
                                inferred.append(output)
                # Add array of predicted outputs for a given question to preds array 
                preds.append(inferred)

        # A sub-function that is to be used with multithreading, judges each prediction using gpt-4o-mini
        def judge_prediction(answer,prediction):
                # Message format for gpt model
                message = [
                        {"role":"system","content":"Evaluate the correctness of the predicted answer"},
                        {"role":"user","content":f"\nTrue answer: {answer}\nPredicted answer: {prediction}"}
                ]
                # Send prompt message to gpt model
                judge_result = apiKey.client.beta.chat.completions.parse(
                        model="gpt-4o-mini",
                        messages=message,
                        response_format=ScoringResult,
                        temperature=0,
                        max_tokens=1024
                )
                # Return parsed result in data format, reason (str) and correct (bool)
                return judge_result.choices[0].message.parsed

        # Number of correct results for each element in dataset 
        correct_count = 0
        # Implement judging using multithreading
        num_workers = os.cpu_count() - 2
        # Run judging function in each available thread
        with ThreadPoolExecutor(max_workers=num_workers) as executor:
                for index, prediction_array in enumerate(preds,start=0):
                        to_judge = {
                                        # Exectute function in thread
                                        executor.submit(judge_prediction,answers[index],p):p
                                        for p in prediction_array
                                    }
                        correct = False
                        # Parse each result as they are completed by each given thread
                        for judged in as_completed(to_judge):
                                    try:
                                        # If at least one is correct, the whole output prediction is correct
                                        if judged.result().correct:
                                                correct = True
                                                break
                                        # Error checking
                                    except Exception as exception:
                                            print(f"Judging generated an exception: {exception}")
                        # Add to total correct count if output prediction was correct
                        if correct:
                                correct_count+=1
                # Get total number of prompts
                total = len(dataset)
                # Percentage of total correct answers
                percentage = correct_count / total * 100 if total > 0 else 0
                return {
                        "Total correct" : correct_count,
                        "Percentage correct" : percentage
                }

# Same as simple eval function, however runs over mutliple samples 
def run_simple_eval_with_multi_sampling(dataset, n_gen=1, m=5, temperature=0.25, top_p=1.0, top_k=2, best_of=1, num_beams=1, max_tokens=512):
        # Total correct array
        correct = []
        # Total percentage of correct passes array
        percentage = []
        for i in range(m):
                # Get result from prompting whole dataset
                result = simple_eval_with_sampling(dataset,n_gen,temperature,top_p,top_k,best_of,num_beams,max_tokens)
                # Add each respective value to the arrays
                correct.append(result["Total correct"])
                percentage.append(result["Percentage correct"])

        # Calculate mean and mean absolute deviation for correct responses
        mean = np.mean(correct)
        mad = 0
        if(len(correct) > 0):
                for c in correct:
                        mad += np.abs(c-mean)
                mad /= len(correct)
                
        # Calculate mean and mean absolute deviation for percentages
        mean_percentage = np.mean(percentage)
        mad_percentage = 0
        if(len(percentage) > 0):
                for p in percentage:
                        mad_percentage += np.abs(p-mean_percentage)
                mad_percentage /= len(percentage)

        return{
                "mean for correct results": round(mean,1),
                "mad for correct results": round(mad,1),
                "mean for percentage results": round(mean_percentage,1),
                "mad for percentage results": round(mad_percentage,1),
        }

# Size of the dataset
dataSize = 2
dataset = datasets.load_dataset("openai/gsm8k","main",split=f"train[0:{dataSize}]",trust_remote_code=True)
trainingDataSet = datasets.load_dataset("openai/gsm8k","main",split=f"train[-{dataSize}:]",trust_remote_code=True)

# Example usage
print(run_simple_eval_with_multi_sampling(dataset, m=3, n_gen=5))