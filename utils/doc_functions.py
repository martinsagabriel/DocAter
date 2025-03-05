import sys
import os
from PyPDF2 import PdfReader

CHUNK_SIZE = 1000
OFFSET = 200

def get_document_path():
    """Get PDF document path from command line arguments.
    
    Returns:
        str: Valid file path provided as first command line argument
        
    Raises:
        SystemExit: If no path provided or file not found
    """ 
    if len(sys.argv) < 2:
        print("Informe o caminho do arquivo\n Exemplo: /caminho/para/arquivo.pdf")
        sys.exit(1)
    else:
        path = sys.argv[1]
    
    if not os.path.exists(path):
        print(f"Erro: O arquivo '{path}' não foi encontrado.")
        sys.exit(1)
    
    return path
    
def document_to_text(document_path):
    """Read a PDF document and return text in string"""
    file = open(document_path, 'rb')
    reader = PdfReader(file)

    document_text = ""
    try:
        for i in range(len(reader.pages)):
            page = reader.pages[i]
            content = page.extract_text()
            document_text += content

        len(document_text)
        return document_text
    except Exception as e:
        print(f"Erro ao transformar o documento '{document_path}': {e}")
        sys.exit(1)

def split_document(document_text, offset=100, chunk_size=1000):
    """Split a document in a list of string"""
    documents = []
    try:
        for i in range(0, len(document_text), chunk_size):
            start = i
            end = i + 1000
            if start != 0:
                start = start - offset
                end =  end - offset
            documents.append(document_text[start:end])
        return documents
    except Exception as e:
        print(f"Erro ao dividir o documento: {e}")
        sys.exit(1)
        

if __name__ == "__main__":
    document_path = get_document_path()
    document = (document_to_text(document_path))
    document_chunks = split_document(document, OFFSET, CHUNK_SIZE)