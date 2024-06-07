from flask import Flask, request
from langchain_community.llms import Ollama
from langchain_community.vectorstores import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings.fastembed import FastEmbedEmbeddings
from langchain_community.document_loaders import PDFPlumberLoader
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain
from langchain.prompts import PromptTemplate

app = Flask( __name__ )

folder_path = "DocumentosDB"
cached_llm = Ollama(model="llama3")

embedding = FastEmbedEmbeddings()

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1024, chunk_overlap=80, length_function=len, is_separator_regex=False
)

#Prompt guía para el modelo
raw_prompt = PromptTemplate.from_template(
    """
    <s>[INST]Eres un asistente experto en buscar información en varios documentos que habla español. Si no tienes una respuesta de la información suministrada indícalo.[/INST] </s>
    [INST] {input} 
            Context: {context}
            Answer:          
    [/INST]                         
    """
)

#Ruta para usar el modelo para responder con base en un query.
@app.route("/ask_pdf", methods=["POST"])
def askPDFPost():
    #Del JSON obtiene el query
    json_content = request.json
    query = json_content["query"]

    #"Creación" o "Recuperación" del contexto
    vector_store = Chroma(persist_directory=folder_path, embedding_function=embedding)

    retriever = vector_store.as_retriever(
        search_type="similarity_score_threshold",
        search_kwargs={
            "k": 10,
            "score_threshold": 0.1
        }
    )

    document_chain = create_stuff_documents_chain(cached_llm, raw_prompt)
    chain = create_retrieval_chain(retriever, document_chain)

    #Con base en el contexto y el query, el modelo genera un resultado
    result = chain.invoke({"input": query})

    #Se recuperan las fuentes usadas
    sources = []
    for doc in result["context"]:
        sources.append(
            {"source": doc.metadata["source"], "page_content": doc.page_content}
        )

    #Se arma la respuesta y se envía la respuesta
    response_answer = {"answer": result["answer"], "sources": sources}
    return response_answer

#Ruta para subir un documento PDF
#Esta ruta va a buscar documentos en ./pdf
@app.route("/pdf", methods=["POST"])
def pdfPost():
    #Abre el archivo
    file = request.files["file"]
    file_name = file.filename
    save_file = "pdf/" + file_name
    file.save(save_file)
    print(f"filename: {file_name}")

    #Divide el archivo (split)
    loader = PDFPlumberLoader(save_file)
    docs = loader.load_and_split()
    print(f"docs len={len(docs)}")

    chunks = text_splitter.split_documents(docs)
    print(f"chunks len={len(chunks)}")

    #Guarda los chucks del archivo con su embeding en Chroma
    vector_store = Chroma.from_documents(
        documents=chunks, embedding=embedding, persist_directory=folder_path
        )
    
    vector_store.persist()

    #Respuesta de confirmación
    response = {
        "status": "Successfully Uploaded", 
        "filename": file_name, 
        "doc_len": len(docs), 
        "chunks": len(chunks)
    }
    return response

def startup_app():
    app.run(host="0.0.0.0", port=8080, debug=True)

if __name__ == "__main__":
    startup_app()