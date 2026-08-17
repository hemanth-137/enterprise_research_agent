from doc_parser import doc_parser
from chunking import process_doc_chunks
#from embedding import create_embeddings
from to_vector_db import add_embedd_to_db
from pathlib import Path
from local_embedding import create_embeddings
import json

def main_ingest():


    db_name = "test_embedd_collection"
    BATCH_SIZE = 32


    # folder = Path("data/pdfs")

    # docling_docs = doc_parser(folder)

    # print("parsed all docs")
    # print("="*60)

    # processed_chunks = process_doc_chunks(docling_docs)

    # print("got all chunks")
    # print("="*60)


    # import json

    # with open("processed_chunks.json", "w", encoding="utf-8") as f:
    #     json.dump(processed_chunks, f, ensure_ascii=False, indent=2)

    # print(f"Saved {len(processed_chunks)} chunks.")

    with open("processed_chunks.json", "r", encoding="utf-8") as f:
        processed_chunks = json.load(f)


    embedded_chunks = create_embeddings(processed_chunks,batch_size=BATCH_SIZE)

    print("got all embeddings")
    print("="*60)

    vector_db_info = add_embedd_to_db(embedded_chunks,db_name=db_name)

    print("embeddings added to vector DB")
    print("="*60)

    print(vector_db_info)

if __name__ == "__main__":
    main_ingest()

    # folder = Path("data/pdfs")
    # print(folder.exists())