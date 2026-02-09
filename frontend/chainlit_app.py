# frontend/chainlit_app.py
"""
Enhanced Chainlit demo UI for Agentic-RAG with improved answer presentation.

Requirements:
  pip install chainlit requests

Run:
  chainlit run frontend/chainlit_app.py --port 8001
"""
import os
import requests
import chainlit as cl

# Configuration
BACKEND_URL = os.getenv("BACKEND_URL", "http://localhost:8000")

def backend_query(query: str, top_k: int = 5, max_tokens: int = 512, temperature: float = 0.0):
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

@cl.on_chat_start
async def start():
    """Welcome message."""
    welcome_msg = """# 👋 Agentic RAG Ready
I am connected and ready. Ask me questions about the uploaded documents.
"""
    await cl.Message(content=welcome_msg).send()

@cl.on_message
async def main(message: cl.Message):
    """Main chat handler."""
    user_query = message.content.strip()
    
    # 1. Show loading state
    msg = cl.Message(content="🔎 **Thinking...**")
    await msg.send()

    # 2. Call Backend (Async wrapper)
    # We use default settings: max_tokens=512, temperature=0
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
            # Try to get text from various possible keys
            meta = s.get("metadata") or s.get("meta") or {}
            text = s.get("text") or meta.get("text") or "No text available"
            # Truncate text for display
            preview = text[:150].replace("\n", " ") + "..."
            source_text += f"{i}. {preview}\n"

    # 5. Send Final Answer
    msg.content = f"{answer}{source_text}"
    await msg.update()

    # 6. Add Action Buttons
    actions = [
        cl.Action(
            name="long_answer", 
            payload={"query": user_query}, 
            label="📝 Long Answer (1024 tokens)"
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
    
    msg = cl.Message(content="📝 **Generating detailed answer...**")
    await msg.send()

    # Call backend with higher token limit
    resp = await cl.make_async(backend_query)(query, max_tokens=1024)

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

    # Call backend with higher temperature
    resp = await cl.make_async(backend_query)(query, temperature=0.7)

    if "error" in resp:
        msg.content = f"❌ Error: {resp['error']}"
    else:
        answer = resp.get("answer", "No answer.")
        msg.content = f"**Creative Answer:**\n\n{answer}"
    
    await msg.update()