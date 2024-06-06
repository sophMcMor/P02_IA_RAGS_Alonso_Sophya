from flask import Flask, request
from langchain_community.llms import Ollama #modelo llama 3
from langchain_community.vectorstores import Chroma #base de datos
from langchain.text_splitter import RecursiveCharacterTextSplitter ##Para el manejo de los pdf
from langchain_community.embeddings.fastembed import FastEmbedEmbeddings
from langchain_community.document_loaders import PDFPlumberLoader
from langchain.chains.combine_documents import create_stuff_documents_chain #Para el retrival 
from langchain.chains import create_retrieval_chain
from langchain.prompts import PromptTemplate


'''
Este es el modelo sencillo, con el que hicimos las pruebas
iniciales. No está entrenado.

'''

app = Flask(__name__)

folder_path = "db"

cached_llm = Ollama(model="llama3")

embedding = FastEmbedEmbeddings()

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1024,
    chunk_overlap=80,
    length_function=len,
    is_separator_regex=False
)

#Este es el prompt que se le va a enviar al modelo para darle un poco de contexto
# raw_prompt = PromptTemplate.from_template(
#     """ 
#     <s>[INST] You are a technical assistant good at searching documents. If you do not have an answer from the provided information say so. [/INST] </s>
#     [INST] {input}
#            Context: {context}
#            Answer:
#     [/INST]
# """
# )

raw_prompt = PromptTemplate.from_template(
    """ 
    <s>[INST] Eres un asistente experto en buscar información documentos, hablas español. Si no tienes una respuesta en la información suministrada indícalo. [/INST] </s>
    [INST] {input}
           Context: {context}
           Answer:
    [/INST]
"""
)

@app.route("/ai", methods=["POST"])
def aiPost():
    print("Post /ai called")
    json_content = request.json
    query = json_content.get("query")

    print(f"query: {query}")

    response = cached_llm.invoke(query)

    print(response)

    response_answer = {"answer": response}
    return response_answer

@app.route("/ask_pdf", methods=["POST"])
def askPDFPost():
    print("Post /ask_pdf called")
    json_content = request.json
    query = json_content.get("query")

    print(f"query: {query}")

    print("Loading vector store")
    vector_store = Chroma(persist_directory=folder_path, embedding_function=embedding)

    #Retriver con búsqueda de similarilidad
    print("Creating chain")
    retriever = vector_store.as_retriever(
        search_type="similarity_score_threshold",
        search_kwargs={
            "k": 20,
            "score_threshold": 0.1,
        },
    )

    document_chain = create_stuff_documents_chain(cached_llm, raw_prompt)
    chain = create_retrieval_chain(retriever, document_chain)

    result = chain.invoke({"input": query})

    print(result)

    sources = []
    for doc in result["context"]:
        sources.append(
            {"source": doc.metadata["source"], "page_content": doc.page_content}
        )

    #response_answer = {"answer": result["answer"], "sources": sources}
    response_answer = {"answer": result["answer"]}
    return response_answer

# @app.route("/pdf", methods=["POST"])
# def pdfPost():
#     print(request.files)
#     file = request.files["file"]
#     file_name = file.filename
#     save_file = "pdf/" + file_name
#     file.save(save_file)
#     print(f"filename: {file_name}")

#     #Cargar el archivo   y convertirlo en "Documentos"
#     loader=PDFPlumberLoader(save_file)
#     docs = loader.load_and_split()
#     print(f"docs len={len(docs)}")

#     #Ya que tenemos el archivo en "documentos", podemos dividirlo en CHUNKS
#     chunks = text_splitter.split_documents(docs)
#     print(f"chunks len={len(chunks)}")
    
#     #Cramos el vector de Chroma DB para guardar chunks
#     vector_store = Chroma.from_documents(
#         documents=chunks, embedding=embedding, persist_directory=folder_path
#     )

#     vector_store.persist()

#     response = {
#         "status": "Successfully Uploaded",
#         "filename": file_name,
#         "doc_len": len(docs),
#         "chunks": len(chunks),
#     }
#     return response

@app.route("/pdf", methods=["POST"])
def pdfPost():
    print(request.files)
    if "file" not in request.files:
        return "No file part", 400
    
    file = request.files["file"]
    if file.filename == "":
        return "No selected file", 400

    file_name = file.filename
    save_file = "pdf/" + file_name
    file.save(save_file)
    print(f"filename: {file_name}")

    # Cargar el archivo y convertirlo en "Documentos"
    loader = PDFPlumberLoader(save_file)
    docs = loader.load_and_split()
    print(f"docs len={len(docs)}")

    # Ya que tenemos el archivo en "documentos", podemos dividirlo en CHUNKS
    chunks = text_splitter.split_documents(docs)
    print(f"chunks len={len(chunks)}")
    
    # Procesar los documentos en lotes más pequeños
    batch_size = 100  # Ajusta el tamaño del lote según sea necesario
    total_chunks = len(chunks)
    vector_store = None

    for i in range(0, total_chunks, batch_size):
        batch_chunks = chunks[i:i + batch_size]
        
        # Crear o actualizar el vector de Chroma DB para guardar los chunks
        if vector_store is None:
            vector_store = Chroma.from_documents(
                documents=batch_chunks, embedding=embedding, persist_directory=folder_path
            )
        else:
            vector_store.add_documents(batch_chunks)
    
    if vector_store:
        vector_store.persist()

    response = {
        "status": "Successfully Uploaded",
        "filename": file_name,
        "doc_len": len(docs),
        "chunks": len(chunks),
    }
    return response

def start_app():
    app.run(host="0.0.0.0", port=8080, debug=True)


if __name__ == "__main__":
    start_app()