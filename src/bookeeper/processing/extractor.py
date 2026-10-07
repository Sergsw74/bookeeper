"""
LangChain structured output schemas and Ollama-based Concept Knowledge Graph extractor.
"""

import json
import logging
from typing import Any, Dict, List, Literal, Optional

import httpx
from pydantic import BaseModel, Field

from bookeeper.config import OllamaSettings
from bookeeper.processing.chunker import TextChunk

logger = logging.getLogger(__name__)


class ConceptNode(BaseModel):
    """An extracted domain entity, theme, idea, or architectural pattern."""

    name: str = Field(description="Canonical concise title of the concept (e.g. 'Event Sourcing').")
    category: Literal[
        "Concept",
        "Architectural Pattern",
        "Theme",
        "Methodology",
        "Tradeoff",
        "Technology",
    ] = Field(
        default="Concept",
        description="Category classification for the node.",
    )
    description: str = Field(
        description="Clear, 1-2 sentence description explaining the concept based on the text."
    )
    aliases: List[str] = Field(
        default_factory=list,
        description="Alternate names or synonyms mentioned.",
    )


class RelationshipEdge(BaseModel):
    """A directed semantic relationship connecting two concepts."""

    source: str = Field(description="Exact name of the source concept.")
    target: str = Field(description="Exact name of the target concept.")
    relation_type: Literal[
        "IMPLEMENTS",
        "EXTENDS",
        "CONTRASTS_WITH",
        "PART_OF",
        "REQUIRES",
        "INFLUENCES",
        "MITIGATES",
    ] = Field(
        default="INFLUENCES",
        description="Type of directed relationship connecting source -> target.",
    )
    explanation: str = Field(
        default="",
        description="Brief justification or context for why this relationship exists.",
    )


class ExtractedGraph(BaseModel):
    """Container for concepts and relationships extracted from a text chunk."""

    concepts: List[ConceptNode] = Field(default_factory=list)
    relationships: List[RelationshipEdge] = Field(default_factory=list)
    summary: str = Field(
        default="",
        description="High-level 1-sentence synopsis of this section.",
    )


EXTRACTION_PROMPT = """You are an expert technical knowledge graph extractor.
Analyze the following book excerpt from '{book_title}' (Chapter: '{chapter_title}').

Identify the core ideas, architectural patterns, methodologies, and concepts discussed.
Formulate relationships between these concepts.

Respond ONLY with valid JSON matching the following schema:
{{
  "summary": "1-sentence overview of the section",
  "concepts": [
    {{
      "name": "Canonical Concept Name",
      "category": "Concept" | "Architectural Pattern" | "Theme" | "Methodology" | "Tradeoff" | "Technology",
      "description": "Short concise description of what this concept is and how it is used.",
      "aliases": ["synonym1", "synonym2"]
    }}
  ],
  "relationships": [
    {{
      "source": "Exact source concept name",
      "target": "Exact target concept name",
      "relation_type": "IMPLEMENTS" | "EXTENDS" | "CONTRASTS_WITH" | "PART_OF" | "REQUIRES" | "INFLUENCES" | "MITIGATES",
      "explanation": "Why these concepts are related"
    }}
  ]
}}

Excerpt text:
\"\"\"
{chunk_text}
\"\"\"
"""


class KnowledgeExtractor:
    """Extractor communicating with local Ollama instance for structured entity extraction."""

    def __init__(self, settings: Optional[OllamaSettings] = None):
        self.settings = settings or OllamaSettings()
        self.api_url = f"{self.settings.base_url.rstrip('/')}/api/chat"

    def extract_from_chunk(self, chunk: TextChunk) -> ExtractedGraph:
        """Extract concepts and edges from a single text chunk via Ollama."""
        prompt = EXTRACTION_PROMPT.format(
            book_title=chunk.book_title,
            chapter_title=chunk.chapter_title,
            chunk_text=chunk.text,
        )

        payload = {
            "model": self.settings.model,
            "messages": [
                {
                    "role": "system",
                    "content": "You are a precise JSON-only structured data extraction assistant.",
                },
                {"role": "user", "content": prompt},
            ],
            "format": "json",
            "stream": False,
            "options": {
                "temperature": self.settings.temperature,
            },
        }

        try:
            with httpx.Client(timeout=self.settings.timeout) as client:
                resp = client.post(self.api_url, json=payload)
                resp.raise_for_status()
                data = resp.json()

            raw_content = data.get("message", {}).get("content", "{}")
            parsed_json = json.loads(raw_content)
            return ExtractedGraph(**parsed_json)
        except Exception as e:
            logger.warning(f"Failed extraction on chunk {chunk.chunk_id}: {e}")
            # Fallback to empty graph on failure
            return ExtractedGraph()
