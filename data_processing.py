import glob
import pdfplumber
import re
import tiktoken

CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200

ITEM_PATTERN = re.compile(
    r'^\s*(ITEM\s+\d+[A-Z]?\.?)\s*(.*)$',
    re.IGNORECASE
)

tokenizer = tiktoken.get_encoding("cl100k_base")


def count_tokens(text: str) -> int:
    return len(tokenizer.encode(text))


class DataProcessor:

    def parse_pdf(self, pdf_path):
        pages = []
        with pdfplumber.open(pdf_path) as pdf:
            for i, page in enumerate(pdf.pages):
                text = page.extract_text()
                if text:
                    pages.append({
                        "page_number": i + 1,
                        "text": text.strip()
                    })
        return pages

    def assign_items(self, pages):
        current_item = "Unknown"
        for page in pages:
            for line in page["text"].splitlines()[:20]:
                match = ITEM_PATTERN.match(line)
                if match:
                    current_item = match.group(1).upper()
                    break
            page["item"] = current_item
        return pages

    def chunk_pages(self, pages, document, source):
        chunks = []
        buffer_text = ""
        buffer_tokens = 0
        buffer_start = None
        current_item = None

        for page in pages:
            page_tokens = count_tokens(page["text"])

            if page["item"] != current_item:
                if buffer_text:
                    chunks.append({
                        "text": buffer_text.strip(),
                        "metadata": {
                            "document": document,
                            "item": current_item,
                            "page_start": buffer_start,
                            "source": source
                        }
                    })
                buffer_text = ""
                buffer_tokens = 0
                buffer_start = page["page_number"]
                current_item = page["item"]

            if buffer_tokens + page_tokens <= CHUNK_SIZE:
                buffer_text += "\n" + page["text"]
                buffer_tokens += page_tokens
            else:
                chunks.append({
                    "text": buffer_text.strip(),
                    "metadata": {
                        "document": document,
                        "item": current_item,
                        "page_start": buffer_start,
                        "source": source
                    }
                })
                buffer_text = page["text"]
                buffer_tokens = page_tokens
                buffer_start = page["page_number"]

        if buffer_text:
            chunks.append({
                "text": buffer_text.strip(),
                "metadata": {
                    "document": document,
                    "item": current_item,
                    "page_start": buffer_start,
                    "source": source
                }
            })

        return chunks

    def prepare_chunks(self, data_folder="data"):
        all_chunks = []
        for pdf in glob.glob(f"{data_folder}/*.pdf"):
            doc = pdf.split("\\")[-1].replace(".pdf", "")
            pages = self.parse_pdf(pdf)
            pages = self.assign_items(pages)
            all_chunks.extend(self.chunk_pages(pages, doc, pdf))
        return all_chunks
