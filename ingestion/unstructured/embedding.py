import google.generativeai as genai
from chunking import process_doc_chunks
from doc_parser import doc_parser
from dotenv import load_dotenv
import os
import numpy as np
import time

load_dotenv()

    # file_path = "./data/pdfs/0070-pdf.pdf"

    # doc_doc = doc_parser(file_path)
    # processed_chunks = process_doc_chunks(doc_doc)
    # print("1. done doc_doc and processed chunks\n")

    # md_text = doc_doc.export_to_markdown()
    # with open("text_embedd1.txt","w",encoding="utf-8") as file:
    #     file.write(md_text)

    # print("done textembedd text 1\n")
    # with open("text_embedd2.txt","w",encoding="utf-8") as file:
    #     file.write(str(processed_chunks))
    # print("done textembedd text 2\n")

def create_embeddings(processed_chunks, batch_size=10):

    gemini_api = os.getenv("GEMINI_API_KEY")

    genai.configure(api_key=gemini_api)

    for i in range(0, len(processed_chunks), batch_size):

        batch = processed_chunks[i:i + batch_size]

        chunk_texts = [chunk["text"] for chunk in batch]

        result = genai.embed_content(
            model="models/gemini-embedding-001",
            content=chunk_texts,
            task_type="retrieval_document",
            output_dimensionality=768
        )

        

        embeddings_matrix = np.array(result["embedding"])

        norms = np.linalg.norm(
            embeddings_matrix,
            axis=1,
            keepdims=True
        )

        normalized_embeddings = embeddings_matrix / norms

        for j, chunk in enumerate(batch):
            chunk["embedding"] = normalized_embeddings[j].tolist()

        time.sleep(10) #to limit API use as am on free tier

        
    return processed_chunks


if __name__ == "__main__":
    pass









# with open("embed_vec.txt","w",encoding="utf-8") as file:

#         for i, chunk in enumerate(processed_chunks):
#                 file.write(chunk)






# print("==================result========================")
# print(result)

# embedding_values = result['embedding']
# embedding_length = len(embedding_values)

# print("==================embedding========================")
# print(f"Normal embedding length: {len(embedding_values)}")
# print(f"Norm of normal embedding: {np.linalg.norm(embedding_values):.6f}")
# print(f"normal embeddings: {embedding_values[:10]}")


# embedding_values_np = np.array(embedding_values)
# normed_embedding = embedding_values_np / np.linalg.norm(embedding_values_np)

# print("==================embedding========================")
# print(f"Normed embedding length: {len(normed_embedding)}")
# print(f"Norm of normed embedding: {np.linalg.norm(normed_embedding):.6f}")
# print(f"norm embeddings: {normed_embedding[:10]}")

# print("==========================================================")
# print(f"Length of embedding: {embedding_length}")