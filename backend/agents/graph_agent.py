from typing import TypedDict, List, Optional
from langchain_core.messages import HumanMessage
from langchain_ollama import ChatOllama
from langgraph.graph import StateGraph, END
import json

# --- STATE DEFINITION ---
class AgentState(TypedDict):
    question: str
    original_question: str
    chat_history: str  
    documents: List[str]
    decision: str
    generation: str
    steps: List[str]
    retry_count: int
    mode: str
    temperature: float  

# --- THE AGENT CLASS ---
class GraphRAGAgent:
    def __init__(self, retrieve_service, model_name="qwen2.5:7b"):
        self.retrieve_service = retrieve_service
        self.model_name = model_name 
        
        # Router and Grader need strict logic (Low Temp)
        self.json_llm = ChatOllama(model=model_name, temperature=0, format="json")
        self.llm = ChatOllama(model=model_name, temperature=0)
        
        self.max_retries = 3

    # --- NODES ---

    def router(self, state: AgentState):
        """Decides flow based on question."""
        print("---ROUTING QUESTION---")
        question = state.get("original_question", state["question"])
        
        # Simple routing prompt
        prompt = f"""You are a router. 
        1. If user asks for info/facts/summary, output 'vectorstore'.
        2. If user says hi/hello/thanks, output 'chitchat'.
        Question: {question}
        Return JSON: {{ "datasource": "vectorstore" | "chitchat" }}"""
        
        try:
            res = self.json_llm.invoke([HumanMessage(content=prompt)])
            decision = json.loads(res.content).get("datasource", "vectorstore")
        except:
            decision = "vectorstore"
        return {"decision": decision, "steps": ["router"]}

    def general_conversation(self, state: AgentState):
        # NEW: Inject history into chitchat
        prompt = f"""
        Previous Chat History:
        {state.get('chat_history', '')}
        
        User: {state['original_question']}
        Reply politely and conversationally."""
        
        writer = ChatOllama(model=self.model_name, temperature=state["temperature"])
        res = writer.invoke([HumanMessage(content=prompt)]).content
        return {"generation": res, "steps": ["general_conversation"]}

    def retrieve(self, state: AgentState):
        """
        Uses the new Hybrid Retrieval (Graph + Vector).
        """
        print(f"---RETRIEVING ({state['mode']})---")
        top_k = 8 if state["mode"] == "detailed" else 5
        try:
            # UPDATED: Use retrieve_hybrid instead of retrieve
            # This returns List[str] directly
            docs = self.retrieve_service.retrieve_hybrid(state["question"], top_k=top_k)
        except Exception as e:
            print(f"Retrieval error: {e}")
            docs = []
            
        return {"documents": docs, "steps": ["retrieve"]}

    def grade_documents(self, state: AgentState):
        print("---GRADING---")
        if not state["documents"]: return {"documents": []}
        
        # Limit grading context to avoid blowing up context window if docs are huge
        doc_txt = "\n\n".join([f"[{i}] {d[:300]}..." for i, d in enumerate(state["documents"])])
        
        prompt = f"""Identify relevant docs for: {state['question']}
        Docs:
        {doc_txt}
        Return JSON {{ "indices": [0, 2...] }} of relevant docs containing ACTUAL content.
        If you are unsure, include the document."""
        
        try:
            res = self.json_llm.invoke([HumanMessage(content=prompt)])
            indices = json.loads(res.content).get("indices", [])
            filtered = [state["documents"][i] for i in indices if i < len(state["documents"])]
        except:
            # If grading fails, keep all documents (fail-safe)
            filtered = state["documents"] 
        return {"documents": filtered, "steps": ["grade_documents"]}

    def transform_query(self, state: AgentState):
        print("---TRANSFORMING QUERY---")
        # NEW: Transform considering history (e.g., "He" -> "Amico")
        prompt = f"""
        Context: {state.get('chat_history', '')}
        User Question: {state['question']}
        
        Rewrite the user question to be standalone and search-friendly. Replace pronouns (he/she/it) with specific names from context if possible.
        Output ONLY the string."""
        
        new_q = self.llm.invoke([HumanMessage(content=prompt)]).content.strip()
        return {"question": new_q, "retry_count": state["retry_count"]+1}

    def generate(self, state: AgentState):
        print(f"---GENERATING ({state['mode'].upper()} | Temp: {state['temperature']})---")
        question = state["original_question"]
        context = "\n\n".join(state["documents"])
        history = state.get("chat_history", "") # <--- Get history
        
        if state["mode"] == "detailed":
            system_prompt = "You are a comprehensive analyst. Write a detailed, in-depth response. Minimum 300 words."
        else:
            system_prompt = "You are a concise assistant. Answer directly and briefly."

        writer = ChatOllama(model=self.model_name, temperature=state["temperature"])

        # NEW: Inject History into Prompt
        prompt = f"""{system_prompt}
        
        Relevant Context (may include Knowledge Graph relationships):
        {context}
        
        Chat History:
        {history}
        
        Question: {question}
        Answer:"""
        
        res = writer.invoke([HumanMessage(content=prompt)]).content
        return {"generation": res, "steps": ["generate"]}

    # --- EDGES & GRAPH ---
    def route_decision(self, state): return state["decision"]
    
    def decide_to_generate(self, state):
        if not state["documents"]:
            return "generate" if state["retry_count"] >= self.max_retries else "transform_query"
        return "generate"

    def build_graph(self):
        workflow = StateGraph(AgentState)
        workflow.add_node("router", self.router)
        workflow.add_node("chitchat", self.general_conversation)
        workflow.add_node("retrieve", self.retrieve)
        workflow.add_node("grade_documents", self.grade_documents)
        workflow.add_node("transform_query", self.transform_query)
        workflow.add_node("generate", self.generate)

        workflow.set_entry_point("router")
        workflow.add_conditional_edges("router", self.route_decision, {"chitchat":"chitchat", "vectorstore":"retrieve"})
        workflow.add_edge("retrieve", "grade_documents")
        workflow.add_conditional_edges("grade_documents", self.decide_to_generate, {"transform_query":"transform_query", "generate":"generate"})
        workflow.add_edge("transform_query", "retrieve")
        workflow.add_edge("chitchat", END)
        workflow.add_edge("generate", END)
        return workflow.compile()

    def query(self, query: str, mode: str = "concise", temperature: float = 0.1, chat_history: str = ""):
        """Entry point that accepts mode, temperature, and chat_history."""
        app = self.build_graph()
        initial = {
            "question": query,
            "original_question": query,
            "chat_history": chat_history,
            "documents": [],
            "decision": "vectorstore",
            "retry_count": 0,
            "steps": [],
            "generation": "",
            "mode": mode,
            "temperature": temperature
        }
        res = app.invoke(initial)
        return {"answer": res["generation"], "metadata": {"steps": res["steps"]}}