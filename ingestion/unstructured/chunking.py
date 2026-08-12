from docling.chunking import HybridChunker
from doc_parser import doc_parser

file_path = "./data/pdfs/0097-pdf.pdf"
docling_doc = doc_parser(file_path)

def doc_chunking(docling_doc:object):

    chunker  = HybridChunker()

    chunk_iter = chunker.chunk(dl_doc = docling_doc)

    for i, chunk in enumerate(chunk_iter):
        print(f"=== {i} ===")
        print(f"chunk.text:\n{f'{chunk.text}'!r}")

        enriched_text = chunker.contextualize(chunk=chunk)
        print(f"chunker.contextualize(chunk):\n{f'{enriched_text}'!r}")

        print()


doc_chunking(docling_doc)
