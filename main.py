import os
import json
import torch
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForCausalLM

from data_processing import DataProcessor
from indexing import FaissIndexer
from qa_engine import QAEngine

# Models
embedding_model = SentenceTransformer("BAAI/bge-base-en-v1.5")

tokenizer = AutoTokenizer.from_pretrained("microsoft/phi-3-mini-4k-instruct")
model = AutoModelForCausalLM.from_pretrained(
    "microsoft/phi-3-mini-4k-instruct",
    dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
    device_map="auto"
)

# Prepare data
processor = DataProcessor()
chunks = processor.prepare_chunks()

indexer = FaissIndexer(embedding_model)
indexer.build_index(chunks)
indexer.save("sec_10k.index", "sec_10k_metadata.pkl")

indexer.load("sec_10k.index", "sec_10k_metadata.pkl")

qa = QAEngine(model, tokenizer)

# Load questions
with open("questions/questions.json", "r") as f:
    questions = json.load(f)

results = []

for q in questions:
    retrieved = indexer.retrieve(q["question"], k=5)
    answer = qa.generate_answer(q["question"], retrieved)
    results.append({
        "question": q["question"],
        "answer": answer["answer"],
        "sources": answer["sources"]
    })

# Save output
os.makedirs("output", exist_ok=True)

with open("output/answers.json", "w") as f:
    json.dump(results, f, indent=2)

print("All answers generated and saved to output/answers.json")
