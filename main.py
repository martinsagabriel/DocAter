from utils.doc_functions import get_document_path, document_to_text, split_document

CHUNK_SIZE = 1000
OFFSET = 200

if __name__ == "__main__":
    document_path = get_document_path()
    document = (document_to_text(document_path))
    document_chunks = split_document(document, OFFSET, CHUNK_SIZE)
    
    print(document_chunks)
