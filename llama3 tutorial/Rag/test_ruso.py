import os
import torch
from flask import Flask, request, jsonify
from unsloth import FastLanguageModel
from datasets import load_dataset
from sentence_transformers import SentenceTransformer
import chromadb

# Cargar el adaptador LoRA
from peft import PeftModel, PeftConfig

app = Flask(__name__)

# Ruta al adaptador LoRA
lora_weights = './adapter_model.safetensors'
lora_config = './adapter_config.json'

# Cargar el modelo Llama3 con Ollama
max_seq_length = 2048
load_in_4bit = True

base_model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="unsloth/llama-3-8b-bnb-4bit",
    max_seq_length=max_seq_length,
    load_in_4bit=load_in_4bit,
)

# Aplicar el adaptador LoRA al modelo base
lora_model = PeftModel.from_pretrained(base_model, lora_config, lora_weights)

# Configuración de ChromaDB
chroma_client = chromadb.Client()
collection = chroma_client.create_collection("pdf_embeddings")

model_embeddings = SentenceTransformer('all-MiniLM-L6-v2')

def get_relevant_context(question, top_k=1):
    question_embedding = model_embeddings.encode([question])
    results = collection.query(
        query_embeddings=question_embedding,
        n_results=top_k
    )
    if results and 'metadatas' in results:
        return results['metadatas'][0]['text']
    return ""

def chatbot(question):
    context = get_relevant_context(question)
    input_text = f"Context: {context}\n\nQuestion: {question}\nAnswer:" if context else f"Question: {question}\nAnswer:"
    inputs = tokenizer(input_text, return_tensors="pt")
    outputs = lora_model.generate(**inputs, max_new_tokens=100)
    answer = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return answer

@app.route('/upload', methods=['POST'])
def upload_file():
    file = request.files['file']
    file.save(os.path.join("./", file.filename))
    doc_text = extract_text_from_pdf(file.filename)
    doc_embedding = model_embeddings.encode(doc_text)
    collection.add(
        ids=[file.filename],
        embeddings=[doc_embedding],
        metadatas=[{"text": doc_text}]
    )
    return jsonify({"message": "File uploaded and processed successfully"})

@app.route('/ask', methods=['POST'])
def ask_question():
    data = request.json
    question = data['question']
    answer = chatbot(question)
    return jsonify({"answer": answer})

def extract_text_from_pdf(pdf_path):
    import fitz
    doc = fitz.open(pdf_path)
    text = ""
    for page in doc:
        text += page.get_text()
    return text

if __name__ == '__main__':
    app.run(debug=True)
