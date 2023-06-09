from flask import Flask, request, jsonify
from langchain import OpenAI
from llama_index import SimpleDirectoryReader, LangchainEmbedding, GPTListIndex,GPTSimpleVectorIndex, PromptHelper
from llama_index import LLMPredictor, ServiceContext
import os

os.environ["OPENAI_API_KEY"] = "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"
app = Flask(__name__)

@app.post("/predict")
def predict():
    input_index = 'index.json'
    index = GPTSimpleVectorIndex.load_from_disk(input_index)
    text = request.get_json().get("message")
    response = index.query(text, response_mode="tree_summarize")
    message = {"answer": response.response}
    return jsonify(message)

if __name__ == "__main__":
    app.run()    
