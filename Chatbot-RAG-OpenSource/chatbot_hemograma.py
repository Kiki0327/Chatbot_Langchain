from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from transformers import pipeline
from langchain.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings

# path vectorstore
path_index = 'G:/Mi unidad/Cursos/Platzi/Chatbot RAG/'
name_index = 'indice-RAG-Hemograma'

# Model embeddings
embedding = HuggingFaceEmbeddings()

# vectorstore
vectorstore = Chroma(name_index, embedding)

# retriever
retriever = vectorstore.as_retriever(search_type="similarity",search_kwargs={'k': 10})

# Model text-generation
llm = pipeline("text-generation", model="meta-llama/Meta-Llama-3-8B")

# Prompt
system_prompt = (
    "Usa el contexto dado para responder a la pregunta."
    "Si no sabes la respuesta, di que no sabes."
    "mantenga la respuesta concisa"
    "Contexto: {context}"
)
prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system_prompt),
        ("human", "{input}"),
    ]
)
question_answer_chain = create_stuff_documents_chain(llm, prompt)
chain = create_retrieval_chain(retriever, question_answer_chain)

# Función para llamar al chat
def chat():
    while True:
        pregunta = input("You: ")
        if pregunta.lower() == "adios":
            break
        
        response = chain.invoke({"input": pregunta})
        print('-'*120)
        print('AI: ' + response)
        print(' ')
        print('='*120)
