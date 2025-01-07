import os
import mimetypes
import logging
import openai
from flask import Flask, request, jsonify
from strategies.chatretrievalstrategy import ChatRetrievalStrategy
from langchain.text_splitter import CharacterTextSplitter
# from langchain.indexes.vectorstore import VectorstoreIndexCreator
from langchain_openai.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores.chroma import Chroma
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_community.document_loaders import PyPDFLoader
from tenacity import (
    retry,
    stop_after_attempt,
    wait_random_exponential,
) # for exponential backoff

from dotenv import load_dotenv, find_dotenv
_ = load_dotenv(find_dotenv()) # read local .env file

openai.api_key  = os.environ['OPENAI_API_KEY']

# Creating VectorDB
@retry(wait=wait_random_exponential(min=1, max=60), stop=stop_after_attempt(6))
def createDB(docs, embeddings, persist_directory):
    vectordb = Chroma.from_documents(documents=docs, embedding=embeddings, persist_directory=persist_directory)
    vectordb.persist()
    return vectordb

sub_dir = 'data'
# Use the below sub dir path when running debug session
# sub_dir = 'Python/OmniGPT/data'

full_path = os.path.abspath(os.path.join(os.path.dirname( os.path.dirname( os.getcwd() )), sub_dir))
loader = PyPDFDirectoryLoader(full_path)
documents = loader.load()
text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
docs = text_splitter.split_documents(documents)
embeddings =  OpenAIEmbeddings(api_key=openai.api_key)
# index = VectorstoreIndexCreator(
#     text_splitter=text_splitter,
#     vectorstore_cls=Chroma
# ).from_loaders([loader])
persist_directory = '../../db' 

vectordb = None
if (os.path.isdir(persist_directory)):
    vectordb = Chroma(persist_directory=persist_directory, embedding_function=embeddings)
else:
    vectordb = createDB(docs=docs, embeddings=embeddings, persist_directory=persist_directory)

# Various strategies to retrieve knowledge, can be extended for additional strategies
chat_strategy = {
    "crs": ChatRetrievalStrategy(vectordb)
}

app = Flask(__name__)

@app.route("/", defaults={"path": "index.html"})
@app.route("/<path:path>")
def static_file(path):
    return app.send_static_file(path)

@app.route("/data/<path>")
def content_file(path):
    mime_type = mimetypes.guess_type(path)
    if mime_type == "application/octet-stream":
        mime_type = mimetypes.guess_type(path)[0] or "application/octet-stream"
    logging.info(path)
    print(path, flush=True)
    loaderPDF = PyPDFLoader(path)
    docs = loaderPDF.load()
    return docs, 200, {"Content-Type": mime_type, "Content-Disposition": f"inline; filename={path}"}

@app.route("/chat", methods=["POST"])
def chat():
    ensure_openai_token()
    retrievalstrategy = request.json["retrievalstrategy"]
    try:
        impl = chat_strategy.get(retrievalstrategy)
        if not impl:
            return jsonify({"error": "unknown retrieval strategy"}), 400
        r = impl.run(request.json["history"], request.json.get("overrides") or {})
        return jsonify(r)
    except Exception as e:
        logging.exception("Exception in /chat")
        return jsonify({"error": str(e)}), 500

def ensure_openai_token():
    openai.api_key  = openai.api_key

if __name__ == '__main__':
    app.run(debug=True)
