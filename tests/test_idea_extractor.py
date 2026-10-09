"""
Unit tests for the expert knowledge-graph idea extractor adjustments.
"""

import json
from unittest.mock import MagicMock, patch
import pytest

from bookeeper.processing.extractor import (
    Concept,
    ExtractedIdea,
    KnowledgeExtractor,
    SectionExtraction,
    validate_concept_chunk_grounding,
)
from bookeeper.processing.ollama_pool import OllamaPool


def test_extracted_idea_schema_and_conversion():
    """Verify ExtractedIdea fields, normalization, and to_concept conversion."""
    idea = ExtractedIdea(
        source_quote="In ant colonies, no single ant understands the global architecture of the nest, yet through simple local pheromone interactions, highly coordinated structures arise spontaneously.",
        idea_statement="Complex coordination emerges from local rules without central leadership.",
        key_entities=["Systems Theory", "Emergence", "Self-organization"],
        idea_type="Mechanism",
    )

    assert idea.source_quote.startswith("In ant colonies")
    assert idea.idea_statement.startswith("Complex coordination")
    assert idea.key_entities == ["Systems Theory", "Emergence", "Self-organization"]
    assert idea.idea_type == "Mechanism"

    concept = idea.to_concept()
    assert isinstance(concept, Concept)
    assert concept.name == "Systems Theory"
    assert concept.category == "Mechanism"
    assert concept.supporting_quote == idea.source_quote
    assert concept.brief_description == idea.idea_statement
    assert "Emergence" in concept.related_concepts


def test_section_extraction_bidirectional_sync():
    """Verify SectionExtraction syncs extracted_ideas <-> concepts seamlessly."""
    # 1. Input with extracted_ideas only
    raw_ideas = {
        "extracted_ideas": [
            {
                "source_quote": "Exact sentence from text.",
                "idea_statement": "Factual statement.",
                "key_entities": ["Topic A", "Topic B"],
                "idea_type": "Definition",
            }
        ]
    }
    ext1 = SectionExtraction.model_validate(raw_ideas)
    assert len(ext1.extracted_ideas) == 1
    assert len(ext1.concepts) == 1
    assert ext1.concepts[0].name == "Topic A"
    assert ext1.concepts[0].supporting_quote == "Exact sentence from text."

    # 2. Input with concepts only
    raw_concepts = {
        "concepts": [
            {
                "name": "Event Sourcing",
                "brief_description": "Store all changes as immutable events.",
                "category": "Architectural Pattern",
                "supporting_quote": "Event sourcing models state changes as events.",
                "related_concepts": ["CQRS"],
            }
        ]
    }
    ext2 = SectionExtraction.model_validate(raw_concepts)
    assert len(ext2.concepts) == 1
    assert len(ext2.extracted_ideas) == 1
    assert ext2.extracted_ideas[0].idea_statement == "Store all changes as immutable events."
    assert ext2.extracted_ideas[0].key_entities == ["Event Sourcing", "CQRS"]


def test_knowledge_extractor_prompt_and_extract_ideas():
    """Verify KnowledgeExtractor formats the exact prompt and parses extracted_ideas."""
    pool = OllamaPool.from_urls(["http://mock-ollama:11434"])
    extractor = KnowledgeExtractor(pool=pool, model="llama3.1:8b")

    chunk_text = (
        "Systems theory demonstrates that complex networks exhibit emergent properties. "
        "In ant colonies, no single ant understands the global architecture of the nest, "
        "yet through simple local pheromone interactions, highly coordinated structures arise spontaneously."
    )

    mock_response = SectionExtraction(
        extracted_ideas=[
            ExtractedIdea(
                source_quote="In ant colonies, no single ant understands the global architecture of the nest, yet through simple local pheromone interactions, highly coordinated structures arise spontaneously.",
                idea_statement="Complex coordination emerges from local rules without central leadership.",
                key_entities=["Systems Theory", "Emergence", "Self-organization"],
                idea_type="Mechanism",
            )
        ]
    )

    with patch.object(extractor, "_execute_structured_invoke", return_value=mock_response) as mock_invoke:
        ideas = extractor.extract_ideas(
            text=chunk_text,
            book_title="Emergence and Complexity",
            section_title="Chapter 1: Ant Colonies",
        )

        assert len(ideas) == 1
        assert ideas[0].idea_type == "Mechanism"
        assert ideas[0].source_quote in chunk_text

        # Verify prompt messages sent to LLM
        call_args = mock_invoke.call_args
        schema_cls = call_args[0][0]
        messages = call_args[0][1]

        assert schema_cls is SectionExtraction
        system_content = messages[0].content
        human_content = messages[1].content

        # Verify exact rules from prompt
        assert "You are an expert knowledge-graph extractor." in system_content
        assert "Rules:" in system_content
        assert "Every idea must be anchored by an exact verbatim quote" in system_content
        assert '"idea_statement" must be a concise, self-contained factual claim' in system_content
        assert "Extract only distinct, meaningful concepts" in system_content
        assert '"extracted_ideas": [' in system_content
        assert '"idea_type": "Definition" | "Argument" | "Mechanism" | "Example"' in system_content

        # Verify chunk structure
        assert "Now process the following text:" in human_content
        assert "Chunk:\n\"\"\"" in human_content
        assert chunk_text in human_content
