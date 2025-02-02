from flask import Flask, render_template, request, jsonify
from flask_cors import CORS
import torch, torchvision
import transformers, accelerate
from transformers import pipeline

torch.cuda.empty_cache()
torch.cuda.reset_max_memory_allocated()
transformers.logging.set_verbosity_debug()

app = Flask(__name__)
CORS(app)

# Load all models
models = {
    #"gpt2": pipeline("text-generation", model="gpt2"),
    "DS-R1-Qwen-1.5B": pipeline("text-generation",model="deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B",device_map="auto",torch_dtype="auto",max_new_tokens=50)
}

# Load inference endpoint HTML for webpage
@app.route("/")
def init():
    return render_template('index.html')    

# Gets available models and necessary data from server
@app.route("/data", methods=["POST"])
def pull():
    values = []
    for name, value in models.items():
        values.append(name)
    return jsonify(values)

@app.route("/infer", methods=["POST"])
def infer():
    # Parse request
    data = request.get_json()
    model_name = data.get("model")
    inputs = data.get("inputs")
    
    # Validate inputs
    if not model_name or model_name not in models:
        return jsonify({"error": "Invalid model name"}), 400
    if not inputs:
        return jsonify({"error": "No inputs provided"}), 400
    
    # Perform inference, returns whole output by model
    model = models[model_name]
    try:
        result = model(inputs)  # Model-specific inference
        return jsonify({"model": model_name, "result": result})
    except Exception as e:
        return jsonify({"error": str(e)}), 500
    
# Run app from host
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
