import tiktoken
from docling_core.transforms.chunker.tokenizer.openai import OpenAITokenizer

from docling_core.transforms.chunker.hybrid_chunker import HybridChunker

from docling_core.transforms.chunker.hierarchical_chunker import (
    ChunkingDocSerializer,
    ChunkingSerializerProvider,
)
from docling_core.transforms.serializer.markdown import MarkdownTableSerializer


CHUNK_SIZE = 500


def extract_text(chunker: HybridChunker, chunk) -> str:

    return chunker.contextualize(chunk=chunk)


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


def process_doc_chunks(docling_doc):

    class MarkdownTableSerializerProvider(ChunkingSerializerProvider):

        def get_serializer(self, doc):
            return ChunkingDocSerializer(
                doc=doc,
                table_serializer=MarkdownTableSerializer(),
            )

    tokenizer = OpenAITokenizer(
        tokenizer=tiktoken.get_encoding("cl100k_base"),
        max_tokens=CHUNK_SIZE,
    )

    chunker = HybridChunker(
        tokenizer=tokenizer,
        merge_peers=True,
        repeat_table_header=True,
        serializer_provider=MarkdownTableSerializerProvider(),
    )

    processed_chunks = []

    for i, chunk in enumerate(chunker.chunk(dl_doc=docling_doc)):
        
        processed_chunks.append({
            "id": f"chunk_{i:03d}",
            "text": extract_text(chunker, chunk),
            "metadata": extract_metadata(chunk),
        })

    return processed_chunks


if __name__ == "__main__":

    from doc_parser import doc_parser

    file_path = "./data/pdfs/0002-pdf.pdf"

    docling_doc = doc_parser(file_path)

    result = process_doc_chunks(docling_doc)

    tokenizer = OpenAITokenizer(
        tokenizer=tiktoken.get_encoding("cl100k_base"),
        max_tokens=CHUNK_SIZE,
    )

    with open("test_chunk_02.txt", "w", encoding="utf-8") as file:

        file.write(f"Number of chunks: {len(result)}\n")

        for chunk in result:
            file.write("\n" + "=" * 80 + "\n")
            file.write(f"{chunk['id']}\n")

            txt_tokens = tokenizer.count_tokens(chunk["text"])
            file.write(f"no.tokens = {txt_tokens}\n")

            file.write(f"metadata = {chunk['metadata']}\n")
            file.write(f"{chunk['text']}\n")