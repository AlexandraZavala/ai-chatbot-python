
from sklearn.metrics.pairwise import cosine_similarity
from pydantic import BaseModel
from langchain_pinecone import PineconeVectorStore
from sentence_transformers import SentenceTransformer
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter, Language
import os
import tempfile
from git import Repo
from openai import OpenAI
from pathlib import Path
from langchain.schema import Document
import PIL.Image
import google.generativeai as genai
from dotenv import load_dotenv
import logging
from fastapi import FastAPI, HTTPException, Query
from pinecone.grpc import PineconeGRPC as Pinecone

load_dotenv()

logger = logging.getLogger(__name__)

genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

SUPPORTED_EXTENSIONS = {'.py', '.js', '.tsx', '.jsx', '.ipynb', '.java',
                         '.cpp', '.ts', '.go', '.rs', '.vue', '.swift', '.c', '.h'}


IGNORED_DIRS = {'node_modules', 'venv', 'env', 'dist', 'build', '.git',
                '__pycache__', '.next', '.vscode', 'vendor'}

app = FastAPI()
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

client = OpenAI(
    base_url="https://api.groq.com/openai/v1",
    api_key=GROQ_API_KEY
)

# Configura Pinecone
pinecone_client = Pinecone(
    api_key=PINECONE_API_KEY, 
    environment="us-east-1"  
)
pinecone_index = pinecone_client.Index("codebase")

def clone_repo(repo_url: str) -> str:
    if not repo_url.startswith("http"):
        raise ValueError("La URL del repositorio no parece ser válida.")
    
    try:
        repo_name = repo_url.split("/")[-1].replace(".git", "")
        temp_dir = tempfile.mkdtemp()
        repo_path = os.path.join(temp_dir, repo_name)
        logger.info(f"Clonando el repositorio {repo_url} en {repo_path}")
        
        Repo.clone_from(repo_url, repo_path)
        logger.info(f"Clonado exitoso en: {repo_path}")
        return repo_path
    except Exception as e:
        logger.error(f"Error al clonar el repositorio: {e}")
        raise

def get_file_content(file_path, repo_path):
  try:
    with open(file_path, "r", encoding="utf-8") as f:
      content = f.read()

      rel_path = os.path.relpath(file_path, repo_path)

      return {
          "name": rel_path, #relative path
          "content": content #the code is here
      }
  except Exception as e:
    print(f"Error reading file {file_path}: {e}")
    return None

def get_main_files_content(repo_path: str):
    """
    Get content of supported code files from the local repository.

    Args:
        repo_path: Path to the local repository

    Returns:
        List of dictionaries containing file names and contents
    """
    files_content = []

    try:
        for root, _, files in os.walk(repo_path):
            # Skip if current directory is in ignored directories
            if any(ignored_dir in root for ignored_dir in IGNORED_DIRS):
                continue

            # Process each file in current directory
            for file in files:
                file_path = os.path.join(root, file)
                if os.path.splitext(file)[1] in SUPPORTED_EXTENSIONS:
                    file_content = get_file_content(file_path, repo_path)
                    if file_content:
                        files_content.append(file_content)

    except Exception as e:
        print(f"Error reading repository: {str(e)}")

    return files_content

def get_huggingface_embeddings(text, model_name="sentence-transformers/all-mpnet-base-v2"):
    model = SentenceTransformer(model_name)
    return model.encode(text)

def insert_codebase_into_pinecone(file_content, index_name, namespace):
    documents = []

    for file in file_content:
    #Document type for langchain

        doc = Document(
            page_content=f"{file['name']}\n{file['content']}",
            metadata={"source": file['name']}
        )
        documents.append(doc)

    rec_char_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000, chunk_overlap=100)
    rec_char_docs = rec_char_splitter.split_documents(documents)
    print(rec_char_docs)

    PineconeVectorStore.from_documents(
        documents=rec_char_docs,  
        embedding=HuggingFaceEmbeddings(),
        index_name=index_name,
        namespace=namespace
    )


def perform_rag(query, namespace):
    # Convert the user's query into a numerical embedding
    raw_query_embedding = get_huggingface_embeddings(query)
    
    # Search for the top 5 most relevant matches in the Pinecone index based on the query embedding
    top_matches = pinecone_index.query(vector=raw_query_embedding.tolist(), top_k=5, include_metadata=True, namespace=namespace)

    # Get the list of retrieved texts
    contexts = [item['metadata']['text'] for item in top_matches['matches']]

    augmented_query = "<CONTEXT>\n" + "\n\n-------\n\n".join(contexts[ : 10]) + "\n-------\n</CONTEXT>\n\n\n\nMY QUESTION:\n" + query

    # Modify the prompt below as need to improve the response quality
    system_prompt = f"""You are a Senior Software Engineer, specializing in TypeScript with 20 years of experience.

    Answer any questions I have about the codebase, based on the code provided. Always consider all of the context provided when forming a response.
    
    Let's think step by step.
    """

    llm_response = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": augmented_query}
        ]
    )

    return llm_response.choices[0].message.content


def chat_response(query, namespace,img_path):
  raw_query_embedding = get_huggingface_embeddings(query)
    
  top_matches = pinecone_index.query(vector=raw_query_embedding.tolist(), top_k=5, include_metadata=True, namespace=namespace)

  # Get the list of retrieved texts
  contexts = [item['metadata']['text'] for item in top_matches['matches']]

  augmented_query = "<CONTEXT>\n" + "\n\n-------\n\n".join(contexts[ : 10]) + "\n-------\n</CONTEXT>\n\n\n\nMY QUESTION:\n" + query

  # Modify the prompt below as need to improve the response quality
  system_prompt = f"""You are a Senior Software Engineer, specializing in TypeScript with 20 years of experience.

  Answer any questions I have about the codebase, based on the code provided. Always consider all of the context provided when forming a response.

  Let's thik step by step.
  """

  img = PIL.Image.open(img_path)
  model = genai.GenerativeModel(model_name="gemini-1.5-flash")

  response = model.generate_content([
    {"text": system_prompt},  # Enviar el system_prompt como un mensaje 'system'
    {"text": augmented_query},  # El texto que incluye el contexto y la pregunta
    {"image": img}       # La imagen codificada en base64
  ])

    # Devolver el contenido de la respuesta del modelo
  return response["text"]


def new_codespace(repo_path):
    path = clone_repo(repo_path)
    file_content = get_main_files_content(path)
    RecursiveCharacterTextSplitter.get_separators_for_language
    insert_codebase_into_pinecone(file_content,"codebase", repo_path)

class RAGQuery(BaseModel):
    query: str
    codebase: str

class pathCode(BaseModel):
    path: str

@app.get("/get_namespaces")
async def get_namespaces():
    try:
       
        index = pinecone_client.Index("codebase")

        
        index_stats = index.describe_index_stats()

        # Extrae los namespaces
        namespaces = list(index_stats.get("namespaces", {}).keys())

        return {"namespaces": namespaces}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error getting namespaces:: {str(e)}")

@app.post("/perform_rag")
async def rag_query(data: RAGQuery ):
    try:
        
        response = perform_rag(data.query, data.codebase)
        return {"query": data.query, "response": response}
    except Exception as e:
        logger.error(f"Error ejecutando perform_rag: {e}")
        raise HTTPException(status_code=500, detail=f"Error performing rag: {str(e)}")

@app.post("/create_namespace")
async def create(data: pathCode):
    try:
        
        response = new_codespace(data.path)
        return {response}
    except Exception as e:
        logger.error(f"Error: new_codespace: {e}")
        raise HTTPException(status_code=500, detail=f"Error creating namespace: {str(e)}")

