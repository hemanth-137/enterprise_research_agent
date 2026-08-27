import os

from docling_core.transforms.chunker.tokenizer.huggingface import HuggingFaceTokenizer
from transformers import AutoTokenizer
from docling_core.transforms.chunker.hybrid_chunker import HybridChunker
from docling_core.transforms.chunker.hierarchical_chunker import (
    ChunkingDocSerializer,
    ChunkingSerializerProvider,
)
from docling_core.transforms.serializer.markdown import MarkdownTableSerializer


MODEL_NAME = "BAAI/bge-base-en-v1.5"
CHUNK_SIZE = 512

def extract_text(chunker: HybridChunker, chunk) -> str:
    return chunker.contextualize(chunk=chunk) # .contextualize add heading on top of chunks beware of tokens

def extract_metadata(chunk) -> dict:
    meta_dict = chunk.meta.export_json_dict()
    origin = meta_dict.get("origin", {})
    filename = origin.get("filename", "unknown_doc")
    headings = meta_dict.get("headings", [])

    pages = set()
    for item in meta_dict.get("doc_items", []):
        for prov in item.get("prov", []):
            page_no = prov.get("page_no")
            if page_no is not None:
                pages.add(page_no)

    return {
        "doc_name": filename,
        "page_no": sorted(pages),
        "headings": headings,
    }

def create_chunker(model_name = MODEL_NAME,chunk_size = CHUNK_SIZE):
    class MarkdownTableSerializerProvider(ChunkingSerializerProvider):
        def get_serializer(self, doc):
            return ChunkingDocSerializer(
                doc=doc,
                table_serializer=MarkdownTableSerializer(),
            )

    try:
        local_tokenizer = AutoTokenizer.from_pretrained(model_name, local_files_only=True) # to avoid repeated HF calls
    except Exception:
        local_tokenizer = AutoTokenizer.from_pretrained(model_name)

    tokenizer = HuggingFaceTokenizer(
        tokenizer=local_tokenizer,
        max_tokens=chunk_size,
    )

    return HybridChunker(
        tokenizer=tokenizer,
        merge_peers=True,
        repeat_table_header=True,
        serializer_provider=MarkdownTableSerializerProvider(),
    )

def process_doc_chunks(docling_docs,model_name = MODEL_NAME,chunk_size = CHUNK_SIZE):

    chunker = create_chunker(model_name,chunk_size)

    for docling_doc in docling_docs:
        for i, chunk in enumerate(chunker.chunk(dl_doc=docling_doc)):
            metad = extract_metadata(chunk)
            doc_name = os.path.splitext(metad.get("doc_name", "unknown"))[0]

            yield {
                "id": f"{doc_name}_c_{i:03d}",
                "text": extract_text(chunker, chunk),
                "metadata": metad,
            }