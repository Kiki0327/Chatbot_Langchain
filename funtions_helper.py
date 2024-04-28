from langchain_community.document_loaders import PyPDFLoader
import os
from langchain_text_splitters import RecursiveCharacterTextSplitter

def load_files(path_files=str, debug=None):
    """
    Cargador de documentos en formato PDF.

    Args:
        path_files: Ruta en donde se encuentran los archivos.
        debug: Booleano, si se quiere imprimir información del objeto

    Returns:
        Una lista de objetos Document.
    """

    docs = os.listdir(path_files)
    docs = [path_files + x for x in docs]

    documents = []

    for i in docs:
        loader = PyPDFLoader(i)
        data = loader.load()
        documents.extend(data)

    if debug == True:
        print(f'El objeto retornado por la función es de tipo: {type(documents)}. Y cada subobjeto es de tipo: {type(documents[0])}')
        print(f'En total se cargaron {len(documents)} paginas')
        print(f'A continuación se muestran como ejemplo las primeras 5 paginas:\n{documents[0:5]}')
        
    return documents

def Splitting(obj=list, chunk_size=int, chunk_overlap=None, debug=None):
    """
    Fragmentar los documentos previamente cargados.

    Args:
        obj: Lista con objetos tipo documents. En otras palabras tiene que ser una lista con las paginas de los documentos cargados.
        chunk_size: Tamaño de los fragmentos o numero de caracteres de los fragmentos.
        chunk_overlap: Superposicion de los fragmentos. Se recomienda que sea entre el 10% y 15% del chunk_size. Por defecto 10%
        debug: Booleano, si se quiere imprimir información del objeto

    Returns:
        Objeto document fragmentado.
    """

    if chunk_overlap == None:
        chunk_overlap = chunk_size*0.1

    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    texts = splitter.split_documents(obj)

    if debug == True:
        print(f'Los fragmentos son de tamaño: {chunk_size} y la superposicion es igual a: {chunk_overlap}')
        print(f'El número de fragmentos total es igual a: {len(texts)}')

    return texts