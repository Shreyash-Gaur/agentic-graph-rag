import os
from langchain_neo4j import Neo4jGraph, Neo4jVector
from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field  # <--- CHANGED THIS
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from backend.core.config import settings
import logging

logger = logging.getLogger("agentic-rag.graph")

class Entities(BaseModel):
    """Identifying information about entities."""
    names: list[str] = Field(
        ...,
        description="All the person, organization, or business entities that appear in the text",
    )

class GraphService:
    def __init__(self):
        self.graph = Neo4jGraph(
            url=settings.NEO4J_URI,
            username=settings.NEO4J_USERNAME,
            password=settings.NEO4J_PASSWORD
        )
        
        self.embeddings = OllamaEmbeddings(
            model=settings.EMBEDDING_MODEL,
            base_url=settings.OLLAMA_BASE_URL
        )
        
        self.llm = ChatOllama(
            model=settings.OLLAMA_MODEL, 
            temperature=0,
            base_url=settings.OLLAMA_BASE_URL
        )

        # Initialize Entity Extraction Chain
        prompt = ChatPromptTemplate.from_messages([
            ("system", "You are extracting organization and person entities from the text."),
            ("human", "Use the given format to extract information from the following input: {question}"),
        ])
        self.entity_chain = prompt | self.llm.with_structured_output(Entities)

    def get_vector_index(self):
        """Returns the Neo4jVector store interface"""
        return Neo4jVector.from_existing_graph(
            self.embeddings,
            url=settings.NEO4J_URI,
            username=settings.NEO4J_USERNAME,
            password=settings.NEO4J_PASSWORD,
            search_type="hybrid",
            node_label="Document",
            text_node_properties=["text"],
            embedding_node_property="embedding"
        )

    def structured_retriever(self, question: str) -> str:
        """
        Runs the Cypher query to find neighbors of entities mentioned in the question.
        """
        result = ""
        try:
            entities = self.entity_chain.invoke({"question": question})
            for entity in entities.names:
                response = self.graph.query(
                    """CALL db.index.fulltext.queryNodes('fulltext_entity_id', $query, {limit:2})
                    YIELD node,score
                    CALL {
                      WITH node
                      MATCH (node)-[r:!MENTIONS]->(neighbor)
                      RETURN node.id + ' - ' + type(r) + ' -> ' + neighbor.id AS output
                      UNION ALL
                      WITH node
                      MATCH (node)<-[r:!MENTIONS]-(neighbor)
                      RETURN neighbor.id + ' - ' + type(r) + ' -> ' +  node.id AS output
                    }
                    RETURN output LIMIT 50
                    """,
                    {"query": entity},
                )
                result += "\n".join([el['output'] for el in response])
        except Exception as e:
            logger.error(f"Error in graph retrieval: {e}")
        return result

    def close(self):
        pass