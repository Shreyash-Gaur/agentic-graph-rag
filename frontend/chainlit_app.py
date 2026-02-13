# frontend/chainlit_app.py
"""
Enhanced Chainlit demo UI for Agentic-RAG with improved answer presentation.
Run:
  chainlit run frontend/chainlit_app.py --port 8001
"""
import os
import requests
import chainlit as cl
import sys
from pathlib import Path

# Add backend to path to import settings
sys.path.append(str(Path(__file__).resolve().parents[1]))
from backend.core.config import settings

# Configuration
BACKEND_URL = os.getenv("BACKEND_URL", "http://localhost:8000")

def backend_query(query: str, top_k: int = settings.TOP_K_RETRIEVAL, max_tokens: int = settings.MAX_TOKENS, temperature: float = 0.0):
    """
    Send a query to the FastAPI backend.
    """
    url = f"{BACKEND_URL}/query"
    payload = {
        "query": query,
        "top_k": top_k,
        "max_tokens": max_tokens,
        "temperature": temperature
    }
    try:
        response = requests.post(url, json=payload, timeout=150)
        response.raise_for_status()
        return response.json()
    except Exception as e:
        return {"error": str(e)}

def upload_file_to_backend(file_element):
    """
    Uploads a file element to the backend /ingest/upload endpoint.
    """
    url = f"{BACKEND_URL}/ingest/upload"
    try:
        # Chainlit files are stored in a temp path
        with open(file_element.path, "rb") as f:
            files = {"file": (file_element.name, f, file_element.mime)}
            response = requests.post(url, files=files, timeout=60)
            response.raise_for_status()
        return True
    except Exception as e:
        print(f"Upload failed: {e}")
        return False

@cl.on_chat_start
async def start():
    """Welcome message."""
    welcome_msg = """# 👋 Agentic RAG Ready
I am connected and ready. Ask me questions about the uploaded documents.
**📎 You can attach PDF or TXT files to upload them!**
"""
    await cl.Message(content=welcome_msg).send()

@cl.on_message
async def main(message: cl.Message):
    """Main chat handler."""
    
    # --- 0. HANDLE FILE UPLOADS ---
    if message.elements:
        # If the user attached files, process them
        uploaded_files = []
        for element in message.elements:
            # Check if it's a file
            if isinstance(element, cl.File):
                success = await cl.make_async(upload_file_to_backend)(element)
                if success:
                    uploaded_files.append(element.name)
        
        if uploaded_files:
            await cl.Message(
                content=f"✅ **Uploaded:** {', '.join(uploaded_files)}\n"
                        f"*The watcher is processing these files. They will be available for search shortly.*"
            ).send()
            
        # If there is no text query with the file, stop here
        if not message.content:
            return

    # --- REGULAR QUERY LOGIC ---
    user_query = message.content.strip()
    
    # 1. Show loading state
    msg = cl.Message(content="🔎 **Thinking...**")
    await msg.send()

    # 2. Call Backend (Async wrapper)
    resp = await cl.make_async(backend_query)(user_query)

    # 3. Handle Errors
    if "error" in resp:
        msg.content = f"❌ **Error:** {resp['error']}"
        await msg.update()
        return

    # 4. Format Output
    answer = resp.get("answer", "No answer generated.")
    sources = resp.get("sources", [])
    
    # Simple source listing
    source_text = ""
    if sources:
        source_text = "\n\n---\n**📚 Sources:**\n"
        for i, s in enumerate(sources, 1):
            meta = s.get("metadata") or s.get("meta") or {}
            text = s.get("text") or meta.get("text") or "No text available"
            preview = text[:150].replace("\n", " ") + "..."
            source_text += f"{i}. {preview}\n"

    # 5. Send Final Answer
    msg.content = f"{answer}{source_text}"
    await msg.update()

    # 6. Add Action Buttons
    base_tokens = settings.MAX_TOKENS
    long_tokens = base_tokens * 2

    actions = [
        cl.Action(
            name="long_answer", 
            payload={"query": user_query, "max_tokens": long_tokens}, 
            label=f"📝 Long Answer ({long_tokens} tokens)"
        ),
        cl.Action(
            name="creative_answer", 
            payload={"query": user_query}, 
            label="🎨 Creative Answer"
        )
    ]
    await cl.Message(content="**Options:**", actions=actions).send()

@cl.action_callback("long_answer")
async def on_long_answer(action):
    """Handler for Long Answer button."""
    query = action.payload["query"]
    req_tokens = action.payload["max_tokens"]
    
    msg = cl.Message(content="📝 **Generating detailed answer...**")
    await msg.send()

    resp = await cl.make_async(backend_query)(query, max_tokens=req_tokens)

    if "error" in resp:
        msg.content = f"❌ Error: {resp['error']}"
    else:
        answer = resp.get("answer", "No answer.")
        msg.content = f"**Detailed Answer:**\n\n{answer}"
    
    await msg.update()

@cl.action_callback("creative_answer")
async def on_creative_answer(action):
    """Handler for Creative Answer button."""
    query = action.payload["query"]
    
    msg = cl.Message(content="🎨 **Thinking creatively...**")
    await msg.send()

    resp = await cl.make_async(backend_query)(query, temperature=0.7)

    if "error" in resp:
        msg.content = f"❌ Error: {resp['error']}"
    else:
        answer = resp.get("answer", "No answer.")
        msg.content = f"**Creative Answer:**\n\n{answer}"
    
    await msg.update()