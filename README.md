# 🧠 Agentic Graph RAG: A Self-Correcting Hybrid Retrieval System

> **A local-first RAG system that combines Structured Knowledge (Graph) with Unstructured Context (Vector) using an autonomous agentic workflow.**

## 📖 Overview

Standard RAG pipelines often fail on complex queries that require **multi-hop reasoning** or **global dataset understanding**. This project solves that by implementing a **Hybrid Agentic Architecture**.

Instead of blindly retrieving top-k chunks, this system employs a **Router-Grader-Solver** loop:

1. **Router:** Dynamically selects the best retrieval strategy (Graph, Vector, or Both).
2. **Grader:** Evaluates retrieved documents for relevance *before* generation.
3. **Self-Correction:** If retrieval is insufficient, the agent rewrites the query and retries.

This approach significantly reduces hallucinations and improves performance on questions requiring relationship traversal (e.g., *Who is X's great-grandfather?*).

---

## 🏗️ Architecture

The system runs on a **dual-pipeline** architecture:

### 1. The Knowledge Graph (Structured)

* **Engine:** Neo4j
* **Role:** Handles relational queries (Family trees, hierarchies, ownership).
* **Ingestion:** Uses LLMs to extract Entities and Relationships from unstructured text.

### 2. The Vector Store (Unstructured)

* **Engine:** FAISS (Local) & Neo4j Vector Index
* **Role:** Handles semantic search for descriptions, summaries, and broad concepts.
* **Ingestion:** Standard chunking and embedding (via Ollama).

### 3. The Reasoning Agent (The Brain)

* **Routing:** Uses an LLM classifier to decide:
* *Relational Query?*  **Execute Cypher Query**
* *Descriptive Query?*  **Execute Vector Search**
* *Complex Query?*  **Sequential Hybrid Search**


* **Memory:** Persists conversational state in SQLite for multi-turn context.

---

## 🚀 Key Features

* **Hybrid Retrieval:** Seamlessly blends Graph traversal with Vector similarity search.
* **Agentic Routing:** Does not hard-code the retrieval path; the system *decides* the best tool for the job.
* **Self-Correction (Reflection):** If the initial search yields poor results, the Agent rewrites the query and tries a different strategy.
* **Precision Re-ranking:** Uses a Cross-Encoder (`ms-marco-MiniLM`) to score and filter retrieved documents, ensuring high relevance before the LLM sees them.
* **Long-Term Memory:** Conversation history is persisted in SQLite, allowing for context-aware follow-up questions.
* **Local-First Privacy:** Built on **Ollama**, ensuring no data leaves your local machine (optional OpenAI support available).

---

## 🛠️ Tech Stack

* **LLM & Embeddings:** Ollama (Mistral, Llama 3)
* **Orchestration:** LangChain, LangGraph
* **Graph Database:** Neo4j (Dockerized)
* **Vector Database:** FAISS (Local Index)
* **Backend:** FastAPI
* **Frontend:** Chainlit
* **Monitoring:** Custom Python Logging

---

## ⚡ Getting Started

### Prerequisites

* Docker & Docker Compose
* Python 3.11+
* Ollama (running locally)

### Installation

1. **Clone the repository**
```bash
git clone https://github.com/yourusername/agentic-graph-rag.git
cd agentic-graph-rag

```


2. **Start Infrastructure (Neo4j)**
```bash
docker-compose up -d

```


3. **Install Dependencies**
```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

```


4. **Configure Environment**
Create a `.env` file in the root directory:
```bash
cp .env.example .env
# Edit .env with your Neo4j credentials and Ollama model names

```


5. **Ingest Data (Automated)**
The system supports continuous ingestion. Run two "watchers" in separate terminals to automatically process any file dropped into the `knowledge/` folder.
**Terminal 1: Vector Watcher (FAISS)**
*Watches for new files and updates the semantic search index.*
```bash
python backend/scripts/ingest_watch.py --watch knowledge

```


**Terminal 2: Graph Watcher (Neo4j)**
*Watches for new files and extracts entities/relationships for the Graph.*
```bash
python backend/scripts/ingest_graph_watch.py --watch knowledge

```


*Now, simply drag-and-drop PDFs into the `knowledge/` folder, and the system will learn them automatically!*
6. **Run the Application**
* **Backend:** `uvicorn backend.main:APP --reload --port 8000`
* **Frontend:** `chainlit run frontend/chainlit_app.py -w --port 8001`



---

## 🧪 Example Scenarios

### Scenario 1: Multi-Hop Reasoning (Graph)

> **User:** *"How is Amico related to Giovanni Caruso?"*
> **System Reasoning:**
> 1. Router detects "related to"  Selects **Graph Tool**.
> 2. Executes Cypher query to traverse 3 generations.
> 3. **Result:** *"Amico is the great-grandson of Giovanni."* (Correctly inferred without explicit text).
> 
> 

### Scenario 2: Semantic Search (Vector)

> **User:** *"Describe the social causes the family supports."*
> **System Reasoning:**
> 1. Router detects "Describe"  Selects **Vector Tool**.
> 2. Retrieves text chunks about "Food for All" and "Charity."
> 3. **Result:** Summarized description of their philanthropic work.
> 
> 

### Scenario 3: Self-Correction (Hybrid)

> **User:** *"List all restaurants and their specific locations."*
> **System Reasoning:**
> 1. Initial Graph Search yields a list of names but misses locations.
> 2. **Grader** marks the answer as "Incomplete."
> 3. **Agent** transforms query to *"Locations of Caruso restaurants"* and triggers **Vector Search**.
> 4. **Result:** Combines Graph names with Vector locations for a complete answer.
> 
> 

---

## 🔮 Future Improvements

* Implement **GraphRAG (Microsoft approach)** for community detection.
* Deploy to AWS/GCP using container orchestration.
* Add multi-modal support (Images/Tables).

---

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](https://www.google.com/search?q=LICENSE) file for details.