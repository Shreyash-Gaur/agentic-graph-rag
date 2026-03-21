"""
GraphService — Neo4j graph traversal and vector search.

Entity disambiguation strategy (three tiers):

  Tier 1 — Better extraction prompt
    The entity_chain uses a system prompt that instructs the LLM to always
    use the most complete, specific form of a name and never merge distinct
    entities that share a partial name (e.g. "M. Hamel" and "C. Hamel" are
    different people even though both are "Hamel").

  Tier 2 — Post-ingestion APOC merge
    merge_duplicate_entities() uses Jaro-Winkler similarity to find and merge
    nodes that are formatting variants of the same entity (e.g. "M Hamel" and
    "M. Hamel"). Threshold 0.92 catches spacing/punctuation variants without
    merging distinct entities that share a last name.

  Tier 3 — LLM coreference resolution before extraction
    resolve_coreferences() rewrites each chunk before entity extraction,
    replacing partial names, pronouns, and aliases with their full canonical
    form. This is the only reliable way to handle "Claude" -> "Anthropic Claude"
    because that disambiguation requires reading context, not string matching.
"""

import logging
from langchain_neo4j import Neo4jGraph, Neo4jVector
from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import HumanMessage
from pydantic import BaseModel, Field
from backend.core.config import settings

logger = logging.getLogger("agentic-rag.graph")


class Entities(BaseModel):
    """Identifying information about entities."""
    names: list[str] = Field(
        ...,
        description=(
            "All person, organization, or business entities that appear in the text. "
            "Always use the most complete and specific form of each name."
        ),
    )


class GraphService:
    def __init__(self):
        self.graph = Neo4jGraph(
            url=settings.NEO4J_URI,
            username=settings.NEO4J_USERNAME,
            password=settings.NEO4J_PASSWORD,
        )

        self.embeddings = OllamaEmbeddings(
            model=settings.EMBEDDING_MODEL,
            base_url=settings.OLLAMA_BASE_URL,
        )

        self.llm = ChatOllama(
            model=settings.OLLAMA_MODEL,
            temperature=0,
            base_url=settings.OLLAMA_BASE_URL,
        )

        # ── TIER 1: Extraction prompt with disambiguation rules ──────────────
        # Instructs the LLM to always use the most specific name form and to
        # keep distinct entities with shared last names or partial names separate.
        extraction_prompt = ChatPromptTemplate.from_messages([
            (
                "system",
                """You are extracting person, organization, and business entities
                from text to build a knowledge graph.

                STRICT RULES — follow these exactly:

                1. Always use the MOST COMPLETE and SPECIFIC form of a name.
                - If someone is called both "M. Hamel" and "Hamel", always write "M. Hamel".
                - If an AI model is called both "Claude" and "Anthropic Claude", write "Anthropic Claude".
                - If a product is called both "Gemini" and "Google Gemini", write "Google Gemini".

                2. NEVER merge two distinct entities just because they share a last name or partial name.
                - "M. Hamel" and "C. Hamel" are DIFFERENT people — keep them separate.
                - "Claude Shannon" and "Anthropic Claude" are DIFFERENT entities — keep them separate.

                3. Preserve initials and prefixes exactly — they are the key disambiguators.
                - "M." and "C." are different initials and must never be collapsed.

                4. If a partial name is genuinely ambiguous (context does not clarify which
                entity it refers to), output the partial name as-is rather than guessing.

                5. Normalise formatting only:
                - Fix spacing: "M hamel" → "M. Hamel"
                - Fix casing: "m. hamel" → "M. Hamel"
                - Do NOT change initials, first names, or any substantive part of the name.""",
            ),
            (
                "human",
                "Extract all entities from the following text: {question}",
            ),
        ])
        self.entity_chain = extraction_prompt | self.llm.with_structured_output(Entities)

    # ── TIER 3: LLM coreference resolution ──────────────────────────────────

    def resolve_coreferences(self, text: str) -> str:
        """
        Rewrites text replacing pronouns, partial names, and aliases with
        their full canonical form before entity extraction runs.

        This is the only reliable way to handle cases like:
          "Claude"     -> "Anthropic Claude"
          "Gemini"     -> "Google Gemini"
          "He said..." -> "M. Hamel said..."

        Because these require reading context to disambiguate — string matching
        alone cannot resolve them.

        Cost: one LLM call per chunk. On a local 7B model this adds ~3-8s per
        chunk during ingestion. Use --skip-coref in the watcher to disable for
        faster ingestion at the cost of less accurate entity extraction.
        """
        prompt = f"""Rewrite the following text replacing all partial names,
        pronouns, and aliases with their full canonical name as established in the text.

        Rules:
        - Replace pronouns (he, she, they, his, her) with the actual person's full name
        - Replace partial names with full names only if the full name is clearly
        established in this same text ("Claude" -> "Anthropic Claude" only if
        "Anthropic Claude" appears in this text)
        - If a partial name is ambiguous or the full form is not in this text, leave it as-is
        - Do NOT change any facts, dates, numbers, or relationships
        - Do NOT add information that is not in the original text
        - Return ONLY the rewritten text, no explanation or preamble

        Text:
        {text}

        Rewritten text:"""

        try:
            result = self.llm.invoke([HumanMessage(content=prompt)])
            rewritten = result.content.strip()
            # Sanity check: if the result is much shorter than the original,
            # the LLM probably summarised instead of rewrote — fall back
            if len(rewritten) < len(text) * 0.5:
                logger.warning(
                    "Coreference resolution returned suspiciously short result "
                    "(%d chars vs %d original) — using original text",
                    len(rewritten), len(text),
                )
                return text
            return rewritten
        except Exception as e:
            logger.error("Coreference resolution failed: %s — using original text", e)
            return text

    # ── TIER 2: APOC entity merge ────────────────────────────────────────────

    def merge_duplicate_entities(self) -> int:
        """
        Uses Neo4j APOC and Jaro-Winkler string similarity to find and merge
        entity nodes that are formatting variants of the same real-world entity.

        Jaro-Winkler weights prefix similarity heavily, which means:
          "M. Hamel" vs "M Hamel"        -> ~0.97  (merged — formatting variant)
          "M. Hamel" vs "C. Hamel"        -> ~0.81  (kept separate — different initial)
          "Anthropic Claude" vs "Claude"  -> ~0.72  (kept separate — too different)

        The 0.92 threshold catches spacing/punctuation/casing variants without
        merging distinct entities that share a last name or partial name.

        Requires APOC — already configured in docker-compose.yaml via:
          NEO4J_PLUGINS: '["apoc"]'

        Returns the number of duplicate pairs found and merged.
        """
        # Step 0: fix any nodes whose id became a StringArray from a previous
        # merge run that used properties:'combine'. This query finds nodes
        # where id is a list and converts them back to their longest string.
        # Uses valueType() — built into Neo4j 5.x, no APOC needed.
        cleanup_query = """
        MATCH (n:__Entity__)
        WHERE n.id IS NOT NULL
          AND valueType(n.id) STARTS WITH 'LIST'
        WITH n,
             reduce(longest = '', x IN n.id |
                 CASE WHEN size(toString(x)) > size(longest)
                      THEN toString(x) ELSE longest END
             ) AS canonical_id
        SET n.id = canonical_id
        RETURN count(n) AS fixed
        """

        # Uses elementId() — id() is deprecated in Neo4j 5.x.
        # Uses valueType() guard to skip any remaining StringArray nodes
        # rather than letting toLower() throw TypeError on them.
        find_query = """
        MATCH (a:__Entity__), (b:__Entity__)
        WHERE elementId(a) < elementId(b)
          AND a.id IS NOT NULL
          AND b.id IS NOT NULL
          AND valueType(a.id) = 'STRING NOT NULL'
          AND valueType(b.id) = 'STRING NOT NULL'
          AND apoc.text.jaroWinklerDistance(toLower(a.id), toLower(b.id)) > 0.92
        RETURN
            elementId(a) AS eid_a,
            elementId(b) AS eid_b,
            a.id          AS name_a,
            b.id          AS name_b,
            apoc.text.jaroWinklerDistance(toLower(a.id), toLower(b.id)) AS score
        ORDER BY score DESC
        """

        # FIX: properties:'overwrite' keeps the canonical node's id as a plain
        # string. 'combine' was merging id values into StringArray which then
        # broke subsequent toLower() calls on those nodes.
        # FIX: merge one pair at a time — bulk merge caused "Node not found"
        # when a node was deleted mid-transaction by an earlier pair in the set.
        merge_one_query = """
        MATCH (a:__Entity__), (b:__Entity__)
        WHERE elementId(a) = $eid_a AND elementId(b) = $eid_b
        WITH
            CASE WHEN size(a.id) >= size(b.id) THEN a ELSE b END AS canonical,
            CASE WHEN size(a.id) >= size(b.id) THEN b ELSE a END AS duplicate
        CALL apoc.refactor.mergeNodes([canonical, duplicate], {
            properties: 'overwrite',
            mergeRels:  true
        })
        YIELD node
        RETURN node.id AS merged_into
        """

        try:
            # Run cleanup first to fix any StringArray ids from previous runs
            try:
                result = self.graph.query(cleanup_query)
                fixed = result[0].get("fixed", 0) if result else 0
                if fixed > 0:
                    logger.info("Fixed %d entity node(s) with StringArray id — converted to string", fixed)
            except Exception as e:
                logger.debug("StringArray cleanup skipped (no arrays found or APOC not ready): %s", e)

            candidates = self.graph.query(find_query)
            if not candidates:
                logger.info("No duplicate entity pairs found — graph is clean")
                return 0

            logger.info("Found %d duplicate entity pair(s) — merging:", len(candidates))
            merged_count = 0

            for row in candidates:
                logger.info(
                    "  '%s' + '%s' (score=%.3f)",
                    row["name_a"], row["name_b"], row["score"],
                )
                try:
                    self.graph.query(
                        merge_one_query,
                        {"eid_a": row["eid_a"], "eid_b": row["eid_b"]},
                    )
                    merged_count += 1
                except Exception as e:
                    # Node may have already been merged by a previous iteration
                    logger.debug(
                        "Skipping pair ('%s', '%s') — likely already merged: %s",
                        row["name_a"], row["name_b"], e,
                    )

            logger.info("Entity merge complete — %d pair(s) merged", merged_count)
            return merged_count

        except Exception as e:
            logger.error("Entity merge failed (is APOC installed?): %s", e)
            return 0

    # ── Neo4j vector store ───────────────────────────────────────────────────

    def index_documents_to_neo4j(self, documents: list) -> None:
        """
        Embeds document chunks and writes them as Document nodes with
        embeddings into Neo4j's vector store.

        This MUST be called during ingestion. add_graph_documents() only writes
        entity nodes and relationship edges — it does not embed document text.
        Without this call, Neo4jVector.from_existing_graph() finds no embeddings
        and vector search silently returns empty results, causing every query to
        fall back to FAISS and defeating the purpose of Neo4j as primary store.

        Neo4jVector.from_documents() handles writing Document nodes AND computing
        and storing their embeddings in one call. The hybrid BM25 + dense search
        index is created automatically under the name 'document_vector_index'.
        """
        try:
            Neo4jVector.from_documents(
                documents,
                self.embeddings,
                url=settings.NEO4J_URI,
                username=settings.NEO4J_USERNAME,
                password=settings.NEO4J_PASSWORD,
                index_name="document_vector_index",
                node_label="Document",
                text_node_property="text",
                embedding_node_property="embedding",
                search_type="hybrid",
            )
            logger.info(
                "Indexed %d document chunks into Neo4j vector store.", len(documents)
            )
        except Exception as e:
            logger.error(
                "Failed to index documents into Neo4j vector store: %s", e
            )
            raise

    def get_vector_index(self):
        """
        Returns the Neo4jVector store interface for hybrid BM25 + dense search.
        Reads from the same 'document_vector_index' that index_documents_to_neo4j
        creates — index_name must match between the two calls.
        """
        return Neo4jVector.from_existing_graph(
            self.embeddings,
            url=settings.NEO4J_URI,
            username=settings.NEO4J_USERNAME,
            password=settings.NEO4J_PASSWORD,
            index_name="document_vector_index",
            search_type="hybrid",
            node_label="Document",
            text_node_properties=["text"],
            embedding_node_property="embedding",
        )

    # ── Structured retrieval ─────────────────────────────────────────────────

    def structured_retriever(self, question: str) -> str:
        """
        Extracts named entities from the question using the Tier 1 prompt,
        then runs Cypher to find their graph neighbours.

        Returns a formatted string of relationship triples, e.g.:
          M. Hamel - TALKED_ABOUT -> French Language
          M. Hamel - TAUGHT -> Franz
        """
        result = ""
        try:
            entities = self.entity_chain.invoke({"question": question})
            if not entities.names:
                logger.info(
                    "No entities extracted from question — skipping graph retrieval"
                )
                return result

            logger.info("Extracted entities for graph retrieval: %s", entities.names)

            for entity in entities.names:
                try:
                    response = self.graph.query(
                        """
                        CALL db.index.fulltext.queryNodes(
                            'fulltext_entity_id', $query, {limit: 2}
                        )
                        YIELD node, score
                        CALL {
                          WITH node
                          MATCH (node)-[r:!MENTIONS]->(neighbor)
                          RETURN node.id + ' - ' + type(r) + ' -> ' + neighbor.id AS output
                          UNION ALL
                          WITH node
                          MATCH (node)<-[r:!MENTIONS]-(neighbor)
                          RETURN neighbor.id + ' - ' + type(r) + ' -> ' + node.id AS output
                        }
                        RETURN output LIMIT 50
                        """,
                        {"query": entity},
                    )
                    if response:
                        result += "\n".join([el["output"] for el in response]) + "\n"
                except Exception as e:
                    # Log per-entity failures — don't swallow silently so the
                    # caller can distinguish "graph found nothing" from "graph errored"
                    logger.warning(
                        "Graph query failed for entity '%s': %s", entity, e
                    )

        except Exception as e:
            logger.error("Entity extraction failed: %s", e)

        return result.strip()

    def close(self):
        pass