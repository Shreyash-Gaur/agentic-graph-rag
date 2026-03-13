# backend/agents/graph_agent.py
"""
Graph-augmented RAG agent.

Fixes vs original:
  1. Graph compiled ONCE in __init__ — was rebuilt on every query() call.
  2. HyDE fix: embed only the hypothetical document, not query + hyde_doc
     concatenated. Original was wrong per the HyDE paper.
  3. _invoke_json: strips markdown code fences before JSON parsing so a
     model wrapping output in ```json``` does not silently fall back.
  4. All bare except: replaced with except Exception:.
  5. query() returns sources so callers get evidence.

What is intentionally NOT changed:
  - Pipeline flow: router -> retrieve -> grade -> transform -> generate.
    Intelligence here comes from Neo4j graph traversal, not a planner loop.
  - retrieve_hybrid stays as-is.
"""

from __future__ import annotations

import json
import logging
from typing import TypedDict, List, Dict, Optional

from langchain_core.messages import HumanMessage, ToolMessage
from langchain_ollama import ChatOllama
from langgraph.graph import StateGraph, END

from backend.core.config import settings
from backend.tools.calculator import calculate
from backend.tools.query_expander import generate_hyde_document

logger = logging.getLogger("agentic-rag.agent")


class AgentState(TypedDict):
    question:          str
    original_question: str
    chat_history:      str
    documents:         List[str]
    decision:          str
    generation:        str
    steps:             List[str]
    retry_count:       int
    mode:              str
    temperature:       float
    max_tokens:        int


class GraphRAGAgent:
    """
    LangGraph RAG agent with Neo4j graph + FAISS vector hybrid retrieval.
    Graph is compiled ONCE at construction time and reused for every request.
    """

    def __init__(self, retrieve_service, model_name: str = settings.OLLAMA_MODEL):
        self.retrieve_service = retrieve_service
        self.model_name       = model_name
        self.max_retries      = settings.MAX_ITERATIONS

        self._json_llm = ChatOllama(model=model_name, temperature=0, format="json")
        self._llm      = ChatOllama(model=model_name, temperature=0)

        # FIX 1: compile once, not on every query() call
        self._app = self._build_graph()
        logger.info("GraphRAGAgent compiled and ready (model=%s)", model_name)

    def _invoke_json(self, prompt: str, fallback: Dict) -> Dict:
        """
        FIX 3: strip markdown fences before parsing.
        Without this, ```json wrapping causes silent fallback to default values,
        breaking grading and routing invisibly.
        """
        try:
            res     = self._json_llm.invoke([HumanMessage(content=prompt)])
            content = res.content.strip()
            if content.startswith("```json"):
                content = content[7:]
            elif content.startswith("```"):
                content = content[3:]
            if content.endswith("```"):
                content = content[:-3]
            return json.loads(content.strip())
        except Exception as e:
            logger.warning("JSON LLM parse failed: %s", e)
            return fallback

    def _writer(self, temperature: float, max_tokens: int) -> ChatOllama:
        return ChatOllama(
            model=self.model_name,
            temperature=temperature,
            num_predict=max_tokens,
        )

    def _router(self, state: AgentState) -> Dict:
        logger.info("--- ROUTER ---")
        question = state.get("original_question", state["question"])
        prompt = (
            f"You are a router.\n"
            f"1. If user asks for info/facts/summary output 'vectorstore'.\n"
            f"2. If user says hi/hello/thanks output 'chitchat'.\n"
            f"Question: {question}\n"
            f'Return JSON: {{"datasource": "vectorstore" | "chitchat"}}'
        )
        result   = self._invoke_json(prompt, {"datasource": "vectorstore"})
        decision = result.get("datasource", "vectorstore")
        return {"decision": decision, "steps": ["router"]}

    def _chitchat(self, state: AgentState) -> Dict:
        logger.info("--- CHITCHAT ---")
        prompt = (
            f"Previous chat:\n{state.get('chat_history', '')}\n\n"
            f"User: {state['original_question']}\n"
            f"Reply politely and conversationally."
        )
        writer = self._writer(state["temperature"], state["max_tokens"])
        reply  = writer.invoke([HumanMessage(content=prompt)]).content
        return {"generation": reply, "steps": ["chitchat"]}

    def _retrieve(self, state: AgentState) -> Dict:
        """
        FIX 2: HyDE embeds the hypothetical document ALONE, not concatenated
        with the original query. Original was: query + hyde_doc which defeats
        the purpose — the embedding ends up anchored to the original query
        rather than the synthetic answer space.
        """
        logger.info("--- RETRIEVE (mode=%s) ---", state["mode"])
        top_k        = settings.TOP_K_RETRIEVAL * 2 if state["mode"] == "detailed" else settings.TOP_K_RETRIEVAL
        search_query = state["question"]

        if settings.USE_HYDE:
            try:
                hyde_doc     = generate_hyde_document(state["question"])
                search_query = hyde_doc   # embed hypothetical doc alone
            except Exception as e:
                logger.warning("HyDE failed, using raw query: %s", e)

        try:
            docs = self.retrieve_service.retrieve_hybrid(search_query, top_k=top_k)
        except Exception as e:
            logger.error("Retrieval error: %s", e)
            docs = []

        return {"documents": docs, "steps": ["retrieve"]}

    def _grade_documents(self, state: AgentState) -> Dict:
        logger.info("--- GRADE DOCUMENTS ---")
        if not state["documents"]:
            return {"documents": []}

        doc_txt = "\n\n".join(
            [f"[{i}] {d[:300]}..." for i, d in enumerate(state["documents"])]
        )
        prompt = (
            f"Identify relevant docs for: {state['question']}\n"
            f"Docs:\n{doc_txt}\n"
            f'Return JSON {{"indices": [0, 2, ...]}} of relevant docs. '
            f"If unsure, include the document."
        )
        result  = self._invoke_json(prompt, {"indices": list(range(len(state["documents"])))})
        indices = result.get("indices", [])
        try:
            filtered = [state["documents"][i] for i in indices if i < len(state["documents"])]
        except Exception:
            filtered = state["documents"]

        return {"documents": filtered, "steps": ["grade_documents"]}

    def _transform_query(self, state: AgentState) -> Dict:
        logger.info("--- TRANSFORM QUERY ---")
        prompt = (
            f"Context: {state.get('chat_history', '')}\n"
            f"User Question: {state['question']}\n\n"
            f"Rewrite to be standalone and search-friendly. "
            f"Replace pronouns with specific names from context if possible.\n"
            f"Output ONLY the rewritten question string."
        )
        new_q = self._llm.invoke([HumanMessage(content=prompt)]).content.strip()
        return {"question": new_q, "retry_count": state["retry_count"] + 1}

    def _generate(self, state: AgentState) -> Dict:
        logger.info("--- GENERATE (mode=%s, tokens=%d) ---", state["mode"], state["max_tokens"])
        question = state["original_question"]
        context  = "\n\n".join(state["documents"])
        history  = state.get("chat_history", "")

        if state["mode"] == "detailed":
            system_prompt = (
                f"You are a comprehensive analyst. Provide a detailed answer "
                f"using up to {state['max_tokens']} tokens. Cover all aspects. "
                f"Do NOT output raw JSON or mention tool names."
            )
        else:
            system_prompt = (
                "You are a concise assistant. Answer directly and briefly. "
                "Do not output JSON."
            )

        prompt = f"""{system_prompt}

Relevant Context:
{context}

Chat History:
{history}

Question: {question}
Answer:"""

        writer         = self._writer(state["temperature"], state["max_tokens"])
        writer_w_tools = writer.bind_tools([calculate])
        messages       = [HumanMessage(content=prompt)]
        response       = writer_w_tools.invoke(messages)

        if response.tool_calls:
            messages.append(response)
            for tc in response.tool_calls:
                if tc["name"] == "calculate":
                    expr = tc["args"].get("expression", "")
                    logger.info("LLM invoked calculator: %s", expr)
                    try:
                        result = calculate.invoke(tc["args"])
                        messages.append(ToolMessage(content=str(result), tool_call_id=tc["id"]))
                    except Exception as e:
                        messages.append(ToolMessage(content=f"Calculation failed: {e}", tool_call_id=tc["id"]))
            response = writer_w_tools.invoke(messages)

        return {"generation": response.content, "steps": state.get("steps", []) + ["generate"]}

    def _route_decision(self, state: AgentState) -> str:
        return state["decision"]

    def _decide_to_generate(self, state: AgentState) -> str:
        if not state["documents"]:
            if state["retry_count"] >= self.max_retries:
                return "generate"
            return "transform_query"
        return "generate"

    def _build_graph(self):
        wf = StateGraph(AgentState)
        wf.add_node("router",          self._router)
        wf.add_node("chitchat",        self._chitchat)
        wf.add_node("retrieve",        self._retrieve)
        wf.add_node("grade_documents", self._grade_documents)
        wf.add_node("transform_query", self._transform_query)
        wf.add_node("generate",        self._generate)

        wf.set_entry_point("router")
        wf.add_conditional_edges(
            "router", self._route_decision,
            {"chitchat": "chitchat", "vectorstore": "retrieve"},
        )
        wf.add_edge("chitchat",       END)
        wf.add_edge("retrieve",       "grade_documents")
        wf.add_conditional_edges(
            "grade_documents", self._decide_to_generate,
            {"transform_query": "transform_query", "generate": "generate"},
        )
        wf.add_edge("transform_query", "retrieve")
        wf.add_edge("generate",        END)
        return wf.compile()

    def query(
        self,
        query:        str,
        mode:         str   = "concise",
        temperature:  float = 0.0,
        max_tokens:   int   = settings.MAX_TOKENS,
        chat_history: str   = "",
    ) -> Dict:
        initial: AgentState = {
            "question":          query,
            "original_question": query,
            "chat_history":      chat_history,
            "documents":         [],
            "decision":          "vectorstore",
            "generation":        "",
            "steps":             [],
            "retry_count":       0,
            "mode":              mode,
            "temperature":       temperature,
            "max_tokens":        max_tokens,
        }
        result = self._app.invoke(initial)
        return {
            "answer":   result.get("generation", ""),
            "sources":  result.get("documents", []),
            "metadata": {"steps": result.get("steps", [])},
        }