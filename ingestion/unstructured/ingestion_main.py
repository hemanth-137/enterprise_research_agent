import os
os.environ["HF_HUB_OFFLINE"] = "1"

from pathlib import Path
import sys

from doc_parser import doc_parser
from chunking import process_doc_chunks
from local_embedding import create_embeddings
from to_vector_db import add_embedd_to_db_stream


def main_ingest(input_path: str, 
                db_name: str,
                chunk_size: int = 512,
                t_model_name: str = "BAAI/bge-base-en-v1.5",
                batch_size_embed: int = 32, 
                batch_size_db: int = 500):

    target_path = Path(input_path)

    if not target_path.exists():
        print(f"The path '{input_path}' does not exist.")
        sys.exit(1)
        
    is_file = target_path.is_file()
    
    if is_file:
        if target_path.suffix.lower() != ".pdf":
            print(f"file input must be a .pdf file (got {target_path.suffix}).")
            sys.exit(1)
        print(f"running for a single PDF file,  target: {target_path.name}")
    else:
        print(f"Running for a folder, target: {target_path.resolve()}")

    print("=" * 60)
    print("Starting ingestion pipeline...")
    print("=" * 60)

    try:
        print("Loading PDF parser ...")
        doc_stream = doc_parser(target_path, single_file=is_file)

        print("Loading Chunker...")
        chunk_stream = process_doc_chunks(doc_stream,t_model_name,chunk_size)

        print(f"Loading Embedder...")
        embedded_stream = create_embeddings(chunk_stream, batch_size=batch_size_embed) # to change embedding model, do it in local_embedding.py 

        print(f"To vector db for every {batch_size_db} batches")
        total_chunks = add_embedd_to_db_stream(
            embedded_chunks_generator=embedded_stream,
            db_name=db_name,
            batch_size=batch_size_db
        )

        print("=" * 60)
        print(f"Ingestion Pipeline finished. Indexed {total_chunks} chunks into '{db_name}'.")
        print("=" * 60)
        
    except Exception as e:
        print(f"Critical Pipeline error: {e}")
        raise e

if __name__ == "__main__":

    main_ingest(
        input_path=r"D:\movies\OpenRAGBench\pdfs",
        db_name="open_ragbench__collection"
    )
    
    # main_ingest(
    #     input_path=r"D:\enterprise_research_agent\enterprise_research_agent\data\pdfs\resume.pdf",
    #     db_name="enterprise_research_collection"
    # )