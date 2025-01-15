# Parallel sampling with a voting verifier
import apiKey, model
import os
from pydantic import BaseModel
from numpy import np
from concurrent.futures import ThreadPoolExecutor, as_completed

# Structured response format
class ScoringResult(BaseModel):
        reason: str
        correct: bool

def simple_eval_with_sampling(dataset, n_gen=1, temperature=0.7, top_p=1.0, min_p=0.0, top_k=-1, best_of=1, num_beams=1,max_tokens=512):
        preds = []
        answers = []

        samplingParameters = model.SamplingParameters(n=n_gen,temperature=temperature,top_p=top_p,top_k=top_k,best_of=best_of,num_beams=num_beams,max_length=max_tokens,do_sample=False if num_beams > 1 else True)
        for row in dataset:
                question = row["question"]
                answers.append(row["answer"])
                messages = [{"role": "user", "content": question}]
                outputs = model.llm.chat(messages,samplingParameters)
                inferred = []
                for output in outputs[0].outputs:
                        response = output.text
                        if response is not None:
                                inferred.append(response)
                preds.append(inferred)

        def judge_prediction(answer,prediction):
                message = [
                        {"role":"system","content":"Evaluate the correctness of the predicted answer"},
                        {"role":"user","content":f"\nTrue answer: {answer}\nPredicted answer: {prediction}"}
                ]
                judge_result = apiKey.client.beta.chat.completions.parse(
                        model="gpt-4o-mini",
                        messages=message,
                        response_format=ScoringResult,
                        temperature=0,
                        max_tokens=1024
                )
                return judge_result.choices[0].message.parsed

        correct_count = 0
        num_workers = os.cpu_count() - 2
        with ThreadPoolExecutor(max_workers=num_workers) as executor:
                for index, answers in enumerate(preds):
                        to_judge = {
                                        executor.submit(judge_prediction,answers[index],p):p
                                        for p in answers
                                    }
                        for judged in as_completed(to_judge):
                                    try:
                                        if judged.result():
                                                correct_count+=1
                                                break
                                    except Exception as exception:
                                            print(f"Judging generated an exception: {exception}")
                total = len(dataset)
                percentage = correct_count / total * 100 if total > 0 else 0
                return {
                        "Total correct" : correct_count,
                        "Percentage correct" : percentage
                }

def run_simple_eval_with_multi_sampling(dataset, n_gen=16, m=5, temperature=0.7, top_p=1.0, min_p=0.0, top_k=-1, best_of=1, use_beam_search=False,max_tokens=512):
        correct = []
        percentage = []
        for i in range(m):
                result = simple_eval_with_sampling(dataset,n_gen,temperature,top_p,min_p,top_k,best_of,use_beam_search,max_tokens)
                correct.append(result["Total correct"])
                correct.append(result["Percentage correct"])

        mean = np.mean(correct)
        mad = np.mean(np.abs(correct-mean))
        
        mean_percentage = np.mean(percentage)
        mad_percentage = np.mean(np.abs(percentage-mean_percentage))

        return{
                "mean for correct results": round(mean,1),
                "mad for correct results": round(mad,1),
                "mean for percentage results": round(mean_percentage,1),
                "mad for percentage results": round(mad_percentage,1),
        }
        
prompts = [
        "hello my name is",
        "the president of the United States is"
]

test_outputs = model.llm.generate(prompts,SamplingParams(temperature=0))
for output in test_outputs:
        print(f"Prompt: {output.prompt!r}, Generated text: {output.outputs[0].text!r}")