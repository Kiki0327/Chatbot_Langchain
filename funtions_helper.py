from langchain_community.document_loaders import PyPDFLoader
import os
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents.base import Document

def load_files(path_files=str, debug=None):
    """
    Cargador de documentos en formato PDF. Ademas, se realiza una limpieza para eliminar saltos de linea en el texto,
    esto se hace pensando en que los fragmentos y embeddings sean mas coherentes y limpios.

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

    clean_pages = []
    for i in documents:
        clean = i.page_content.replace('-\n', '').replace('\n', '').replace('   ', '')
        clean_page = Document(page_content=clean, metadata=i.metadata)
        clean_pages.append(clean_page)

    if debug == True:
        print(f'El objeto retornado por la función es de tipo: {type(clean_pages)}. Y cada subobjeto es de tipo: {type(clean_pages[0])}')
        print(f'En total se cargaron {len(clean_pages)} paginas')
        print(f'A continuación se muestran como ejemplo las primeras 5 paginas:\n{clean_pages[0:5]}')
        
    return clean_pages

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
        chunk_overlap = chunk_size*0.15

    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    texts = splitter.split_documents(obj)

    if debug == True:
        print(f'Los fragmentos son de tamaño: {chunk_size} y la superposicion es igual a: {chunk_overlap}')
        print(f'El número de fragmentos total es igual a: {len(texts)}')

    return texts