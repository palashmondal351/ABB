import torch


class QAEngine:
    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer
        self.model.eval()

    def trim_context(self, chunks, max_context_tokens):
        selected = []
        total_tokens = 0

        for c in chunks:
            text = c.get("text", "").strip()
            if not text:
                continue

            tokens = len(
                self.tokenizer.encode(text, add_special_tokens=False)
            )

            # Always keep at least ONE chunk
            if total_tokens + tokens > max_context_tokens:
                if not selected:
                    selected.append(c)
                break

            selected.append(c)
            total_tokens += tokens

        return selected

    def format_source(self, metadata):
        doc = metadata.get("document", "Unknown")
        item = metadata.get("item", "Unknown")
        page = metadata.get("page") or metadata.get("page_start") or "NA"

        return [doc, item, f"p.{page}"]

    def build_context(self, chunks):
        lines = []

        for i, ch in enumerate(chunks, 1):
            text = ch.get("text", "")
            metadata = ch.get("metadata", {})

            source = self.format_source(metadata)

            lines.append(
                f"[{i}] {text}\nSource: {source}"
            )

        return "\n\n".join(lines)

    def build_prompt(self, question, context):
        return f"""
You are a financial and legal analysis assistant.

Rules:
1. Use ONLY the information in the Sources.
2. Cite every fact.
3. If not found, say: "Not specified in the document."

Sources:
{context}

Question:
{question}

Answer:
""".strip()

    def generate_answer(self, question, retrieved_chunks):
        print("[QA] Trimming context...")

        trimmed = self.trim_context(retrieved_chunks, max_context_tokens=1800)
        print(f"[QA] Using {len(trimmed)} chunks")

        if not trimmed:
            return {
                "answer": "Not specified in the document.",
                "sources": []
            }

        context = self.build_context(trimmed)
        prompt = self.build_prompt(question, context)

        print("[QA] Tokenizing...")
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=1024
        )

        # IMPORTANT: move inputs to model device safely
        inputs = {k: v.to(self.model.device) for k, v in inputs.items()}

        print("[QA] Starting generation... (CPU may take time)")

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=120,
                do_sample=False,
                eos_token_id=self.tokenizer.eos_token_id,
                pad_token_id=self.tokenizer.eos_token_id
            )

        print("[QA] Generation finished")

        decoded = self.tokenizer.decode(
            outputs[0],
            skip_special_tokens=True
        )

        final_answer = decoded.split("Answer:")[-1].strip()

        sources = [self.format_source(c["metadata"]) for c in trimmed]

        return {
            "answer": final_answer,
            "sources": sources
        }
