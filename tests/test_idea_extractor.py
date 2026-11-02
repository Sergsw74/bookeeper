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


def test_formula_model_and_extraction_heuristics():
    """Verify ExtractedFormula model, LaTeX heuristics, and equation parsing."""
    from bookeeper.processing.extractor import (
        ExtractedFormula,
        extract_raw_formulas_from_text,
        normalize_latex_formula,
        split_formula_sides,
    )

    sample_text = (
        "We define the option temporal-difference objective as:\n"
        "LQh,ϕω =E (st,st+k,g)∼D [ Qh(st,g,ϕω(st,st+k)) − Rh(st:t+k,g) − γVh(st+k,g) ]^2\n\n"
        "In addition, the standard GAN minimax game satisfies:\n"
        "$$V(D, G) = \\mathbb{E}_{x \\sim p_{data}} [\\log D(x)] + \\mathbb{E}_{z \\sim p_z} [\\log (1 - D(G(z)))]$$\n\n"
        "and inline formulation $y = \\mathbf{w}^T \\mathbf{x} + b$ serves as regression baseline."
    )

    formulas = extract_raw_formulas_from_text(sample_text)
    assert len(formulas) >= 3

    # Display math
    gan_f = next((f for f in formulas if "V(D, G)" in f.raw_latex or "p_{data}" in f.raw_latex), None)
    assert gan_f is not None
    assert gan_f.has_formula is True
    assert gan_f.lhs.startswith("V(D, G)")

    # Equation line
    bellman_f = next((f for f in formulas if "LQh" in f.raw_latex), None)
    assert bellman_f is not None
    assert bellman_f.has_formula is True
    assert "LQh" in bellman_f.lhs

    # Inline math
    inline_f = next((f for f in formulas if "\\mathbf{w}^T" in f.raw_latex), None)
    assert inline_f is not None
    assert inline_f.has_formula is True
    assert inline_f.lhs == "y"


def test_reconcile_formulas_and_ideas_deduplication():
    """Verify raw formula preference: LLM duplicate formula idea is dropped, label enriched, raw formula preserved."""
    from bookeeper.processing.extractor import (
        ExtractedFormula,
        ExtractedIdea,
        reconcile_formulas_and_ideas,
    )

    chunk_text = (
        "We define the option temporal-difference objective as:\n"
        "LQh,ϕω =E (st,st+k,g)∼D [ Qh(st,g,ϕω(st,st+k)) − Rh(st:t+k,g) − γVh(st+k,g) ]^2\n"
        "Empirically, premature commitment in answer-first format halves reasoning performance."
    )

    raw_formula = ExtractedFormula(
        raw_latex="LQh,ϕω =E (st,st+k,g)∼D [ Qh(st,g,ϕω(st,st+k)) − Rh(st:t+k,g) − γVh(st+k,g) ]^2",
        lhs="LQh,ϕω",
        supporting_quote="LQh,ϕω =E (st,st+k,g)∼D [ Qh(st,g,ϕω(st,st+k)) − Rh(st:t+k,g) − γVh(st+k,g) ]^2",
        has_formula=True,
    )

    llm_ideas = [
        # LLM idea that is purely a formula duplicate
        ExtractedIdea(
            source_quote="LQh,ϕω =E (st,st+k,g)∼D [ Qh(st,g,ϕω(st,st+k)) − Rh(st:t+k,g) − γVh(st+k,g) ]^2",
            idea_statement="Hierarchical Bellman option loss measures temporal difference over intervals.",
            key_entities=["Hierarchical Option Bellman Loss"],
            idea_type="Mathematical Formulation",
            has_formula=True,
        ),
        # Genuine conceptual idea from chunk
        ExtractedIdea(
            source_quote="Empirically, premature commitment in answer-first format halves reasoning performance.",
            idea_statement="Answer-first structured schemas cause premature commitment.",
            key_entities=["Premature Commitment", "Answer-First Format"],
            idea_type="Empirical Finding",
            has_formula=False,
        ),
    ]

    reconciled_formulas, reconciled_ideas = reconcile_formulas_and_ideas(
        chunk_text=chunk_text,
        raw_formulas=[raw_formula],
        ideas=llm_ideas,
    )

    # Both ideas retained: mathematical formulation concept is kept with has_formula=True, not dropped!
    assert len(reconciled_ideas) == 2
    assert reconciled_ideas[0].has_formula is True
    assert reconciled_ideas[0].idea_type == "Mathematical Formulation"
    assert reconciled_ideas[1].idea_type == "Empirical Finding"
    assert reconciled_ideas[1].key_entities[0] == "Premature Commitment"

    # Raw formula preserved with enriched label from LLM concept
    assert len(reconciled_formulas) == 1
    assert reconciled_formulas[0].raw_latex == raw_formula.raw_latex
    assert reconciled_formulas[0].label == "Hierarchical Option Bellman Loss"
    assert reconciled_formulas[0].has_formula is True


def test_reconcile_formulas_conceptual_idea_retained():
    """Verify conceptual idea referencing a formula is kept and receives has_formula=True."""
    from bookeeper.processing.extractor import (
        ExtractedFormula,
        ExtractedIdea,
        reconcile_formulas_and_ideas,
    )

    raw_formula = ExtractedFormula(
        raw_latex="L(\\theta) = \\sum_i (y_i - f(x_i; \\theta))^2",
        lhs="L(\\theta)",
        supporting_quote="L(\\theta) = \\sum_i (y_i - f(x_i; \\theta))^2",
    )

    llm_ideas = [
        ExtractedIdea(
            source_quote="Using L(\\theta) = \\sum_i (y_i - f(x_i; \\theta))^2, the optimization guarantees convergence.",
            idea_statement="Convex loss guarantees global convergence under Lipschitz continuity.",
            key_entities=["Convex Loss Convergence"],
            idea_type="Theoretical Claim",
            has_formula=False,
        )
    ]

    reconciled_formulas, reconciled_ideas = reconcile_formulas_and_ideas(
        chunk_text="Using L(\\theta) = ...",
        raw_formulas=[raw_formula],
        ideas=llm_ideas,
    )

    assert len(reconciled_formulas) == 1
    assert len(reconciled_ideas) == 1
    assert reconciled_ideas[0].has_formula is True


def test_graph_store_formula_entity_and_edges():
    """Verify Formula is its own type of node in ConceptGraphStore with proper chunk/concept edges."""
    from bookeeper.graph.store import ConceptGraphStore
    from bookeeper.processing.chunker import HierarchicalChunk
    from bookeeper.processing.extractor import Concept, ExtractedFormula

    store = ConceptGraphStore()

    chunk = HierarchicalChunk(
        chunk_id="chk_math_1",
        book_id=42,
        book_title="Hierarchical RL",
        chapter_idx=2,
        chunk_idx=1,
        section_title="Bellman Formulations",
        text="Option loss formulation: LQh = E[...] minimizes error.",
        has_formula=True,
    )

    formula = ExtractedFormula(
        raw_latex="LQh = E_{(s, s')} [(Q - R)^2]",
        normalized_latex="LQh = E_{(s, s')} [(Q - R)^2]",
        label="Hierarchical Bellman Loss",
        lhs="LQh",
        supporting_quote="Option loss formulation: LQh = E[...] minimizes error.",
    )

    concept = Concept(
        name="Hierarchical Reinforcement Learning",
        brief_description="Decomposes tasks into semi-Markov options.",
        has_formula=True,
    )

    # 1. Add chunk and concept
    store.add_chunk(chunk)
    store.add_concept(concept)

    # Verify has_formula attribute on Concept and Chunk nodes
    chunk_node = store.graph.nodes["chunk:chk_math_1"]
    assert chunk_node["has_formula"] is True

    concept_node = store.graph.nodes["concept:Hierarchical Reinforcement Learning"]
    assert concept_node["has_formula"] is True

    # 2. Add formula as its own node type
    f_node_id = store.add_formula(formula, chunk=chunk, concept_name=concept.name)
    assert f_node_id.startswith("formula:")

    f_data = store.graph.nodes[f_node_id]
    assert f_data["type"] == "Formula"
    assert f_data["tag"] == "formula"
    assert f_data["is_formula"] is True
    assert f_data["is_idea"] is False
    assert f_data["has_formula"] is True
    assert f_data["label"] == "Hierarchical Bellman Loss"

    # Verify edges: Chunk <-> Formula
    assert store.graph.has_edge("chunk:chk_math_1", f_node_id)
    assert store.graph["chunk:chk_math_1"][f_node_id]["relation"] == "CONTAINS_FORMULA"
    assert store.graph.has_edge(f_node_id, "chunk:chk_math_1")
    assert store.graph[f_node_id]["chunk:chk_math_1"]["relation"] == "APPEARS_IN"

    # Verify edges: Concept <-> Formula
    assert store.graph.has_edge("concept:Hierarchical Reinforcement Learning", f_node_id)
    assert store.graph["concept:Hierarchical Reinforcement Learning"][f_node_id]["relation"] == "HAS_FORMULA"
    assert store.graph.has_edge(f_node_id, "concept:Hierarchical Reinforcement Learning")
    assert store.graph[f_node_id]["concept:Hierarchical Reinforcement Learning"]["relation"] == "FORMULATES_IDEA"

    # Verify lookup helpers
    retrieved = store.get_formula(f_node_id)
    assert retrieved is not None
    assert retrieved["raw_latex"] == formula.raw_latex

    chunk_formulas = store.get_chunk_formulas("chk_math_1")
    assert len(chunk_formulas) == 1
    assert chunk_formulas[0]["label"] == "Hierarchical Bellman Loss"


def test_extract_section_formula_pre_extraction_and_deduplication():
    """Verify extract_section automatically pre-extracts formulas, reconciles with LLM output, and populates extraction.formulas."""
    pool = OllamaPool.from_urls(["http://mock-ollama:11434"])
    extractor = KnowledgeExtractor(pool=pool, model="llama3.1:8b")

    chunk_text = (
        "We define the multi-step temporal-difference loss as:\n"
        "LQh,ϕω =E (st,st+k,g)∼D [ Qh(st,g,ϕω(st,st+k)) − Rh(st:t+k,g) − γVh(st+k,g) ]^2\n"
        "Answer-first formats degrade reasoning accuracy."
    )

    # LLM extracts the formula as an idea and an empirical finding
    mock_response = SectionExtraction(
        extracted_ideas=[
            ExtractedIdea(
                source_quote="LQh,ϕω =E (st,st+k,g)∼D [ Qh(st,g,ϕω(st,st+k)) − Rh(st:t+k,g) − γVh(st+k,g) ]^2",
                idea_statement="Option Bellman loss measures k-step error.",
                key_entities=["Option Bellman Loss"],
                idea_type="Mathematical Formulation",
            ),
            ExtractedIdea(
                source_quote="Answer-first formats degrade reasoning accuracy.",
                idea_statement="Constrained output formats cause premature commitment.",
                key_entities=["Format Sensitivity"],
                idea_type="Empirical Finding",
            ),
        ]
    )

    with patch.object(extractor, "_execute_structured_invoke", return_value=mock_response):
        res = extractor.extract_section(
            text=chunk_text,
            book_title="Hierarchical Options",
            section_title="Option Objectives",
        )

        # 1. Formulas populated from raw text
        assert len(res.formulas) == 1
        assert "LQh" in res.formulas[0].raw_latex
        assert res.formulas[0].has_formula is True
        assert res.formulas[0].label == "Option Bellman Loss"

        # 2. Both ideas retained: mathematical formulation concept is kept with has_formula=True, not dropped!
        assert len(res.extracted_ideas) == 2
        assert res.extracted_ideas[0].has_formula is True
        assert res.extracted_ideas[0].idea_type == "Mathematical Formulation"
        assert res.extracted_ideas[1].idea_type == "Empirical Finding"
        assert res.extracted_ideas[1].key_entities == ["Format Sensitivity"]

        # 3. Concepts reflect reconciled ideas
        assert len(res.concepts) == 2
        assert res.concepts[0].has_formula is True
        assert res.concepts[0].name == "Option Bellman Loss"
        assert res.concepts[1].name == "Format Sensitivity"


def test_extracted_idea_generic_entity_avoidance():
    """Verify to_concept avoids generic umbrella keywords when descriptive alternatives exist."""
    # Case 1: First entity is generic umbrella 'Model', second is specific 'Attention Sink'
    idea1 = ExtractedIdea(
        source_quote="The model preserves attention sinks at the initial token positions.",
        idea_statement="Initial tokens act as attention sinks to stabilize streaming autoregression.",
        key_entities=["Model", "Attention Sink", "Streaming LLM"],
        idea_type="Mechanism",
    )
    concept1 = idea1.to_concept()
    assert concept1.name == "Attention Sink"
    assert "Model" in concept1.related_concepts
    assert "Streaming LLM" in concept1.related_concepts

    # Case 2: Only a generic umbrella word is present -> enriched with idea_type
    idea2 = ExtractedIdea(
        source_quote="The baseline system processes inputs sequentially.",
        idea_statement="Sequential processing baseline.",
        key_entities=["Baseline"],
        idea_type="Definition",
    )
    concept2 = idea2.to_concept()
    assert concept2.name == "Baseline (Definition)"

    # Case 3: Legitimate multi-word non-umbrella concept
    idea3 = ExtractedIdea(
        source_quote="Mutagenic LUMO screening selects active compounds.",
        idea_statement="LUMO energy levels below -1.5 eV correlate with mutagenicity.",
        key_entities=["Mutagenic LUMO-Threshold Rule", "Molecular Screening"],
        idea_type="Empirical Finding",
    )
    concept3 = idea3.to_concept()
    assert concept3.name == "Mutagenic LUMO-Threshold Rule"


def test_extract_section_formula_promotion_when_llm_empty():
    """Verify that pure math/equation chunks promote formulas to Concepts even when LLM returns empty ideas."""
    pool = OllamaPool.from_urls(["http://mock-ollama:11434"])
    extractor = KnowledgeExtractor(pool=pool, model="llama3.1:8b")

    chunk_text = (
        "We establish the spectral radius contraction bound:\n"
        "$$\\rho(P) < 1$$\n"
        "for all transition operators satisfying positive drift."
    )

    # LLM failed to extract any ideas (common on pure math / code chunks)
    empty_mock_response = SectionExtraction(extracted_ideas=[], concepts=[])

    with patch.object(extractor, "_execute_structured_invoke", return_value=empty_mock_response):
        res = extractor.extract_section(
            text=chunk_text,
            book_title="Markov Contraction",
            section_title="Spectral Radius Bounds",
        )

        # 1. Formulas pre-extracted from LaTeX
        assert len(res.formulas) >= 1
        assert any("\\rho" in f.raw_latex for f in res.formulas)

        # 2. Substantive formula was promoted to extracted_ideas and concepts
        assert len(res.extracted_ideas) >= 1
        assert len(res.concepts) >= 1
        assert res.concepts[0].has_formula is True
        assert res.concepts[0].category == "Mathematical Formulation"
        assert "\\rho(P)" in res.concepts[0].brief_description or "\\rho" in res.concepts[0].name


def test_deduplicator_formula_and_category_guards():
    """Verify deduplicator guards prevent merging distinct formulas or distinct categories with identical labels."""
    from bookeeper.processing.deduplicator import ConceptDeduplicator

    dedup = ConceptDeduplicator(similarity_threshold=0.85, high_similarity_threshold=0.95)

    # Case A: Two different mathematical formulas that share the same name 'Loss'
    c1 = Concept(
        name="Loss",
        brief_description="Bellman error loss",
        category="Mathematical Formulation",
        supporting_quote="L_Q = E[(Q(s, a) - r - gamma V(s'))^2]",
        has_formula=True,
    )
    c2 = Concept(
        name="Loss",
        brief_description="Policy gradient surrogate objective",
        category="Mathematical Formulation",
        supporting_quote="L_CLIP = E[min(r_t * A_t, clip(r_t) * A_t)]",
        has_formula=True,
    )

    is_same, _, method, _, _ = dedup.is_same_concept(c1, c2)
    assert is_same is False
    assert method == "distinct_formula_guard"

    # Case B: Identical name but Mathematical Formulation vs Empirical Finding
    c3 = Concept(
        name="Convergence",
        brief_description="Proof of asymptotic convergence",
        category="Mathematical Formulation",
        supporting_quote="lim_{t -> inf} ||theta_t - theta*|| = 0",
        has_formula=True,
    )
    c4 = Concept(
        name="Convergence",
        brief_description="Empirical convergence rate drops under high learning rate",
        category="Empirical Finding",
        supporting_quote="Empirically, convergence fails when eta exceeds 0.05.",
        has_formula=False,
    )

    is_same_cat, _, method_cat, _, _ = dedup.is_same_concept(c3, c4)
    assert is_same_cat is False
    assert method_cat == "distinct_category_guard"

