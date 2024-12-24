# app.py
# server for calling the model 


import torch
from transformers import AutoModelForSequenceClassification
from transformers import AutoTokenizer
import numpy as np

class SentimentPredictor:

    # initialize, should give the path of model
    def __init__(self, model_dir):
        self.model = AutoModelForSequenceClassification.from_pretrained(model_dir)
        self.tokenizer = AutoTokenizer.from_pretrained(model_dir)

    # give the text and return the prob. of all sentiments
    def predict(self, text):
        inputs = self.tokenizer(text, return_tensors='pt', padding=True, truncation=True, max_length=512)
        with torch.no_grad():
            outputs = self.model(**inputs)
            probabilities = torch.softmax(outputs.logits, dim=-1).numpy()[0]
            predicted_label = ['negative', 'neutral', 'positive'][np.argmax(probabilities)]
        return predicted_label, probabilities
    

from flask import Flask, request, jsonify, render_template
from flask_cors import CORS


CUSTOM_MODEL_PATH = "/path/to/the/model"

app = Flask(__name__)
CORS(app)


predictor = SentimentPredictor(CUSTOM_MODEL_PATH)

@app.route('/')
def home():
    return render_template('index.html')

# call the model and return the result
@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    text = data.get("text")
    label, probabilities = predictor.predict(text)
    result = {
        "label": label,
        "probabilities": {
            "negative": float(probabilities[0]),
            "neutral": float(probabilities[1]),
            "positive": float(probabilities[2])
        }
    }
    print(jsonify(result))
    return jsonify(result)

if __name__ == '__main__':
    app.run(debug=True)

