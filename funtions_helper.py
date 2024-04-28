from langchain.document_loaders import PYPDFloader

def load_files(path_files:list):
    """
    Cargador de documentos en formato PDF.

    Args:
        path_files: Ruta en donde se encuentran los archivos.

    Returns:
        Una lista de objetos Document.
    """
    documents = []
    for i in path_files:
        loader = PYPDFloader(i)
        data = loader.load()
        documents.extend(data)
    
    return documents