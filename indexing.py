import faiss
import numpy as np
import pickle


class FaissIndexer:

    def __init__(self, embedding_model):
        self.embedding_model = embedding_model
        self.index = faiss.IndexFlatIP(768)
        self.texts = []
        self.metadatas = []

    def build_index(self, chunks):
        self.texts = [c["text"] for c in chunks]
        self.metadatas = [c["metadata"] for c in chunks]

        embeddings = self.embedding_model.encode(
            self.texts,
            normalize_embeddings=True,
            show_progress_bar=True
        ).astype("float32")

        self.index.add(embeddings)

    def save(self, index_path, meta_path):
        faiss.write_index(self.index, index_path)
        with open(meta_path, "wb") as f:
            pickle.dump(
                {"texts": self.texts, "metadatas": self.metadatas}, f
            )

    def load(self, index_path, meta_path):
        self.index = faiss.read_index(index_path)
        with open(meta_path, "rb") as f:
            data = pickle.load(f)
            self.texts = data["texts"]
            self.metadatas = data["metadatas"]

    def retrieve(self, query, k=5):
        emb = self.embedding_model.encode(
            "Represent this question for retrieving supporting documents: " + query,
            normalize_embeddings=True
        ).astype("float32")

        scores, idxs = self.index.search(np.array([emb]), k)

        return [
            {
                "text": self.texts[i],
                "metadata": self.metadatas[i]
            }
            for i in idxs[0]
        ]
