import os
import sys
from pathlib import Path
from langchain_community.document_loaders import TextLoader, PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_experimental.graph_transformers import LLMGraphTransformer
from langchain_neo4j import Neo4jGraph
from langchain_ollama import ChatOllama
from dotenv import load_dotenv

# Add backend to path
sys.path.append(str(Path(__file__).resolve().parents[2]))
from backend.core.config import settings

load_dotenv()

def ingest_file(file_path: str):
    print(f"--- Ingesting {file_path} into Neo4j ---")
    
    # 1. Load & Split
    if file_path.endswith(".pdf"):
        loader = PyPDFLoader(file_path)
    else:
        loader = TextLoader(file_path)
        
    docs = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=250, chunk_overlap=24)
    documents = text_splitter.split_documents(documents=docs)
    print(f"Created {len(documents)} chunks.")

    # 2. Initialize Neo4j & LLM
    graph = Neo4jGraph(
        url=settings.NEO4J_URI,
        username=settings.NEO4J_USERNAME,
        password=settings.NEO4J_PASSWORD
    )
    
    llm = ChatOllama(model=settings.OLLAMA_MODEL, temperature=0)
    llm_transformer = LLMGraphTransformer(llm=llm)

    # 3. Convert to Graph Documents
    print("Extracting graph data (this may take time)...")
    graph_documents = llm_transformer.convert_to_graph_documents(documents)
    
    # 4. Add to Neo4j
    print("Saving to Neo4j...")
    graph.add_graph_documents(
        graph_documents,
        baseEntityLabel=True,
        include_source=True
    )
    
    # 5. Create Fulltext Index (Important for the structured retriever!)
    print("Ensuring Fulltext Index...")
    try:
        graph.query(
            "CREATE FULLTEXT INDEX fulltext_entity_id IF NOT EXISTS FOR (n:__Entity__) ON EACH [n.id]"
        )
    except Exception as e:
        print(f"Index creation note: {e}")

    print("--- Ingestion Complete ---")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python backend/scripts/ingest_graph.py <path_to_file>")
    else:
        ingest_file(sys.argv[1])