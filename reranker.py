from sentence_transformers import CrossEncoder


class CrossEncoderReranker:
    def __init__(self, model_name="cross-encoder/ms-marco-MiniLM-L-6-v2"):
        self.model = CrossEncoder(model_name)

    def rerank(self, query, chunks, top_k=5):
        pairs = [(query, c["text"]) for c in chunks]

        scores = self.model.predict(pairs)

        scored_chunks = list(zip(scores, chunks))
        scored_chunks.sort(key=lambda x: x[0], reverse=True)

        return [chunk for _, chunk in scored_chunks[:top_k]]
