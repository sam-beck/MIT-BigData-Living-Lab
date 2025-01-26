from flask import Flask, render_template, request, jsonify
from transformers import pipeline

app = Flask(__name__)

# Load all models
models = {
    "gpt2": pipeline("text-generation", model="gpt2"),
    #"bert-qa": pipeline("question-answering", model="deepset/bert-base-cased-squad2"),
    #"t5-summarization": pipeline("summarization", model="t5-small"),
}

# Load inference endpoint HTML for webpage
@app.route("/")
def init():
    # Load index.html for webpage endpoint
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
