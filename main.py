from utils.doc_functions import get_document_path, document_to_text, split_document

import ollama
import chromadb

CHUNK_SIZE = 1000
OFFSET = 200

if __name__ == "__main__":
    document_path = get_document_path()
    document = (document_to_text(document_path))
    document_chunks = split_document(document, OFFSET, CHUNK_SIZE)
    
    print(document_chunks)
    
    client = chromadb.Client()
    collection = client.create_collection(name="docs")

    # store each document in a vector embedding database
    for i, d in enumerate(document_chunks):
        response = ollama.embed(model="mxbai-embed-large", input=d)
        embeddings = response["embeddings"]
        collection.add(
            ids=[str(i)],
            embeddings=embeddings,
            documents=[d]
        )
