"""
LangChain structured output schemas and Ollama extraction chains for book metadata and concepts.
"""

import json
import logging
import re
import shutil
import subprocess
import time
import urllib.request
from typing import Any, Callable, Dict, List, Optional, Sequence, Set, Tuple

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_ollama import ChatOllama
from pydantic import BaseModel, Field, PrivateAttr, model_validator

from bookeeper.calibre.parser import BookParser
from bookeeper.config import Settings
from bookeeper.processing.filters import has_mathematical_formula
from bookeeper.processing.ollama_pool import OllamaPool

logger = logging.getLogger(__name__)


# ==============================================================================
# Strict Pydantic Output Schemas
# ==============================================================================


class BookMetadata(BaseModel):
    """Cleaned and normalized book catalog metadata."""

    title: str = Field(
        description="Clean canonical book title in original/canonical language (e.g. 'Метро 2033', 'Designing Data-Intensive Applications')."
    )
    author: str = Field(
        description="Canonical author name or comma-separated authors (e.g. 'Дмитрий Глуховский', 'Сергей Зубков', 'Martin Kleppmann')."
    )
    summary: str = Field(
        description="Concise catalog blurb (2-4 sentences) summarizing the book's core premise, subject, or technical themes."
    )

    @model_validator(mode="before")
    @classmethod
    def sanitize_metadata(cls, data: Any) -> Any:
        if isinstance(data, dict):
            for fld in ("title", "author", "summary"):
                if fld in data and isinstance(data[fld], str):
                    data[fld] = BookParser.repair_mojibake(data[fld])
        return data


class Concept(BaseModel):
    """Canonical domain idea, architectural pattern, principle, or technical theme."""

    name: str = Field(
        description="Canonical concise title of the idea or concept (2-4 words, e.g. 'Event Sourcing', 'Leader Election', 'Write-Ahead Logging')."
    )
    brief_description: str = Field(
        default="",
        description="Brief 1-2 sentence description summarizing the core idea.",
    )
    detailed_explanation: str = Field(
        default="",
        description="Detailed multi-sentence explanation of the idea, how it works, mechanisms, and key nuances.",
    )
    category: str = Field(
        default="Idea",
        description="Category classification (e.g. 'Architectural Pattern', 'System Design Principle', 'Theory', 'Tradeoff', 'Data Structure').",
    )
    supporting_quote: Optional[str] = Field(
        default=None,
        description="Direct concise quote or excerpt from the text that directly introduces or supports this idea.",
    )
    related_concepts: List[str] = Field(
        default_factory=list,
        description="List of related idea/concept names discussed in connection with this idea.",
    )
    weight: int = Field(
        default=5,
        ge=0,
        le=10,
        description=(
            "Significance, depth, and specificity weight from 0 to 10. "
            "0-2: very basic/generic concepts (e.g. 'Identification', 'Storage', 'Process', weight=0). "
            "3-5: less generic, valuable foundational concepts (e.g. 'Risk Assessment', weight=3; 'Load Balancing', weight=4). "
            "6-8: specific, advanced architectural approaches or paradigms (e.g. 'RAG', weight=7; 'Event Sourcing', weight=7). "
            "9-10: highly specific, novel, or deep architectural/philosophical concepts (e.g. 'SmartRAG', weight=9; 'Deterministic Simulation Testing', weight=10)."
        ),
    )
    has_formula: bool = Field(
        default=False,
        description="Whether this concept encapsulates or contains a mathematical formula.",
    )
    _embedding: Optional[List[float]] = PrivateAttr(default=None)

    @property
    def embedding(self) -> Optional[List[float]]:
        """Cached vector embedding for this idea/concept (excluded from LLM schema)."""
        return self._embedding

    @embedding.setter
    def embedding(self, val: Optional[List[float]]) -> None:
        self._embedding = [float(x) for x in val] if val is not None else None

    @model_validator(mode="before")
    @classmethod
    def populate_descriptions(cls, data: Any) -> Any:
        if isinstance(data, dict):
            for k in ("name", "brief_description", "detailed_explanation", "category", "supporting_quote", "summary"):
                if k in data and isinstance(data[k], str):
                    data[k] = BookParser.repair_mojibake(data[k])
            if "related_concepts" in data and isinstance(data["related_concepts"], list):
                data["related_concepts"] = [
                    BookParser.repair_mojibake(c) if isinstance(c, str) else c
                    for c in data["related_concepts"]
                ]

            # If 'summary' was provided instead of brief/detailed descriptions
            if "summary" in data and not data.get("brief_description"):
                data["brief_description"] = data["summary"]
            if not data.get("detailed_explanation") and data.get("brief_description"):
                data["detailed_explanation"] = data["brief_description"]
            if not data.get("brief_description") and data.get("detailed_explanation"):
                data["brief_description"] = data["detailed_explanation"]
            if "weight" in data:
                try:
                    w = int(data["weight"])
                    data["weight"] = max(0, min(10, w))
                except (ValueError, TypeError):
                    data["weight"] = 5
        return data

    @property
    def summary(self) -> str:
        """Backward-compatible summary property."""
        return self.brief_description or self.detailed_explanation

    @summary.setter
    def summary(self, val: str) -> None:
        self.brief_description = val

    @property
    def idea_type(self) -> str:
        """Backward-compatible alias for category / idea_type."""
        return self.category

    @property
    def source_quote(self) -> Optional[str]:
        """Backward-compatible alias for supporting_quote."""
        return self.supporting_quote


class ExtractedIdea(BaseModel):
    """
    Core intellectual idea extracted from a book chunk, anchored by an exact verbatim quote.
    """

    source_quote: str = Field(
        description="Exact verbatim sentence from the chunk anchoring this idea.",
    )
    idea_statement: str = Field(
        description="Concise, self-contained factual claim (1–2 sentences).",
    )
    key_entities: List[str] = Field(
        default_factory=list,
        description="List of distinct key entities, topics, or concepts involved.",
    )
    idea_type: str = Field(
        default="Definition",
        description="Type classification: 'Definition' | 'Argument' | 'Mechanism' | 'Example' | 'Formula' | 'Mathematical Formulation' | 'Theoretical Claim' | 'Empirical Finding' | 'Protocol'.",
    )
    has_formula: bool = Field(
        default=False,
        description="Whether this idea encapsulates or contains a mathematical formula.",
    )

    @model_validator(mode="before")
    @classmethod
    def sanitize_idea(cls, data: Any) -> Any:
        if isinstance(data, dict):
            for k in ("source_quote", "idea_statement", "idea_type"):
                if k in data and isinstance(data[k], str):
                    data[k] = BookParser.repair_mojibake(data[k])
            if "key_entities" in data and isinstance(data["key_entities"], list):
                data["key_entities"] = [
                    BookParser.repair_mojibake(e) if isinstance(e, str) else e
                    for e in data["key_entities"]
                ]
            if "has_formula" in data:
                data["has_formula"] = bool(data["has_formula"])
            if "idea_type" in data and isinstance(data["idea_type"], str):
                raw_type = data["idea_type"].strip()
                val = raw_type.title()
                valid_types = {
                    "Definition",
                    "Argument",
                    "Mechanism",
                    "Example",
                    "Formula",
                    "Mathematical Formulation",
                    "Theoretical Claim",
                    "Empirical Finding",
                    "Protocol",
                }
                if val in valid_types:
                    data["idea_type"] = val
                    if val in ("Formula", "Mathematical Formulation"):
                        data["has_formula"] = True
                elif any(m in raw_type.lower() for m in ("math", "formul", "loss", "equation", "objective")):
                    data["idea_type"] = "Formula" if "formul" in raw_type.lower() else "Mathematical Formulation"
                    data["has_formula"] = True
                elif any(m in raw_type.lower() for m in ("theor", "proof", "lemma", "theorem", "bound")):
                    data["idea_type"] = "Theoretical Claim"
                elif any(m in raw_type.lower() for m in ("empir", "find", "result", "ablation", "collapse")):
                    data["idea_type"] = "Empirical Finding"
                elif any(m in raw_type.lower() for m in ("proto", "rule", "gate", "policy", "invariant")):
                    data["idea_type"] = "Protocol"
                elif any(m in raw_type.lower() for m in ("mech", "algo", "loop", "pipeline")):
                    data["idea_type"] = "Mechanism"
        return data

    def to_concept(self) -> Concept:
        """Convert extracted idea into canonical Concept representation for knowledge graph."""
        generic_umbrella_words = {
            "model", "framework", "method", "approach", "algorithm", "architecture",
            "paper", "system", "loss", "gradient", "experiment", "evaluation",
            "benchmark", "dataset", "baseline", "results", "analysis",
        }
        name = ""
        # 1. Inspect key_entities: avoid picking a bare generic umbrella word as concept title
        if self.key_entities:
            first_candidate = self.key_entities[0].strip()
            # If the first candidate is in the generic umbrella list, try to find a better entity
            if first_candidate.lower() in generic_umbrella_words:
                for alt in self.key_entities[1:]:
                    alt_clean = alt.strip()
                    if alt_clean.lower() not in generic_umbrella_words and len(alt_clean) >= 3:
                        name = alt_clean
                        break
            if not name:
                name = first_candidate

        if not name:
            name = self.idea_statement[:60].strip()

        # If name is still just a bare generic umbrella word, enrich it with idea_type
        if name.strip().lower() in generic_umbrella_words:
            name = f"{name} ({self.idea_type})"

        related = [e for e in self.key_entities if e != name]
        high_weight_types = {
            "Formula",
            "Mathematical Formulation",
            "Theoretical Claim",
            "Empirical Finding",
            "Mechanism",
            "Protocol",
            "Argument",
        }
        w = 8 if (self.has_formula or self.idea_type in {"Formula", "Mathematical Formulation", "Theoretical Claim"}) else (7 if self.idea_type in high_weight_types else 5)
        return Concept(
            name=name,
            brief_description=self.idea_statement,
            detailed_explanation=self.idea_statement,
            category=self.idea_type,
            supporting_quote=self.source_quote,
            related_concepts=related,
            weight=w,
            has_formula=self.has_formula,
        )


class ExtractedFormula(BaseModel):
    """
    Mathematical formula entity extracted verbatim from text using LaTeX/math heuristics.
    Formula is a distinct entity type from Idea/Concept.
    """

    raw_latex: str = Field(
        description="Exact raw LaTeX, equation line, or mathematical formulation extracted from the chunk.",
    )
    normalized_latex: str = Field(
        default="",
        description="Cleaned, normalized representation of the mathematical formula.",
    )
    label: str = Field(
        default="",
        description="Label, equation number, or conceptual name associated with the formula (e.g. 'Option Bellman Loss', 'Eq. 1').",
    )
    lhs: str = Field(
        default="",
        description="Left-hand side of equation if applicable.",
    )
    rhs: str = Field(
        default="",
        description="Right-hand side of equation if applicable.",
    )
    supporting_quote: str = Field(
        default="",
        description="Verbatim sentence or passage in the chunk containing the formula.",
    )
    has_formula: bool = Field(
        default=True,
        description="Always True for formula entities.",
    )

    @model_validator(mode="before")
    @classmethod
    def sanitize_formula(cls, data: Any) -> Any:
        if isinstance(data, dict):
            for k in ("raw_latex", "normalized_latex", "label", "lhs", "rhs", "supporting_quote"):
                if k in data and isinstance(data[k], str):
                    data[k] = BookParser.repair_mojibake(data[k])
            raw = data.get("raw_latex", "")
            if not data.get("normalized_latex") and raw:
                data["normalized_latex"] = normalize_latex_formula(raw)
            if not data.get("lhs") and raw:
                lhs, rhs = split_formula_sides(raw)
                if lhs:
                    data["lhs"] = lhs
                if rhs:
                    data["rhs"] = rhs
            data["has_formula"] = True
        return data


# ==============================================================================
# LaTeX and Mathematical Formula Heuristics & Reconciler
# ==============================================================================

DISPLAY_MATH_REGEX = re.compile(
    r"(?:"
    r"\$\$(.+?)\$\$|"
    r"\\\[(.+?)\\\]|"
    r"\\begin\{(?:equation\*?|align\*?|gather\*?|multline\*?)\}(.+?)\\end\{(?:equation\*?|align\*?|gather\*?|multline\*?)\}"
    r")",
    re.DOTALL,
)

INLINE_MATH_REGEX = re.compile(
    r"(?:\$([^\$\n]{3,250})\$|\\\(([^\)\n]{3,250})\\\))"
)

EQUATION_LINE_REGEX = re.compile(
    r"^[ \t]*([\w,\(\)\{\}\\\^\*\[\]\s\-]{1,50}\s*(?:=|\\approx|\\sim|\\le|\\ge|<=|>=|==|≤|≥|≠|≈|∼|∝|∈)\s*.{3,300})[ \t]*$",
    re.MULTILINE,
)

MATH_SYMBOL_REGEX = re.compile(
    r"(?:\\[a-zA-Z]+|[_\^][a-zA-Z0-9{}]|[=≤≥≠≈∼∝∈∉∀∃∑∫∏∂∇√αβγδεθλμστφχψωΓΔΘΛΞΠΣΥΦΨΩ]|\b(?:tr|Cov|Var|argmax|argmin|diag|det)\b|\bE\s*[\[(])"
)


def normalize_latex_formula(raw: str) -> str:
    """Normalize LaTeX formula string by stripping outer delimiters and excessive whitespace."""
    if not raw:
        return ""
    text = raw.strip()
    if text.startswith("$$") and text.endswith("$$") and len(text) >= 4:
        text = text[2:-2].strip()
    elif text.startswith("\\[") and text.endswith("\\]") and len(text) >= 4:
        text = text[2:-2].strip()
    elif text.startswith("$") and text.endswith("$") and len(text) >= 2:
        text = text[1:-1].strip()
    elif text.startswith("\\(") and text.endswith("\\)") and len(text) >= 4:
        text = text[2:-2].strip()
    # Collapse multiple whitespace/newlines
    text = re.sub(r"\s+", " ", text).strip()
    # Strip trailing punctuation often found in text equations
    text = re.sub(r"[,;.]\s*$", "", text)
    return text


def split_formula_sides(raw: str) -> Tuple[str, str]:
    """Split formula into left-hand side (LHS) and right-hand side (RHS) across relation operators."""
    norm = normalize_latex_formula(raw)
    for op in ("=", "\\approx", "\\sim", "\\le", "\\ge", "≤", "≥", "≠", "≈", "∼", "∝", "∈", "==", "<=", ">="):
        if op in norm:
            parts = norm.split(op, 1)
            return parts[0].strip(), parts[1].strip()
    return "", ""


def extract_raw_formulas_from_text(text: str) -> List[ExtractedFormula]:
    """
    Verbatim extraction of mathematical formulas directly from chunk text using LaTeX and equation heuristics.
    Anchor each formula to its exact supporting quote/context in the chunk.
    """
    if not text:
        return []

    extracted: List[ExtractedFormula] = []
    seen_formulas: Set[str] = set()

    # 1. LaTeX display math ($$ ... $$, \[ ... \], \begin{equation} ... \end{equation})
    for m in DISPLAY_MATH_REGEX.finditer(text):
        content = next((g for g in m.groups() if g is not None), "").strip()
        norm = normalize_latex_formula(content)
        if norm and len(norm) >= 3 and norm not in seen_formulas:
            seen_formulas.add(norm)
            lhs, rhs = split_formula_sides(norm)
            quote = m.group(0).strip()
            extracted.append(
                ExtractedFormula(
                    raw_latex=content,
                    normalized_latex=norm,
                    lhs=lhs,
                    rhs=rhs,
                    supporting_quote=quote,
                    has_formula=True,
                )
            )

    # 2. LaTeX inline math ($ ... $, \( ... \))
    for m in INLINE_MATH_REGEX.finditer(text):
        content = next((g for g in m.groups() if g is not None), "").strip()
        norm = normalize_latex_formula(content)
        if norm and len(norm) >= 3 and MATH_SYMBOL_REGEX.search(norm):
            if not any(norm in prev.normalized_latex or prev.normalized_latex in norm for prev in extracted):
                seen_formulas.add(norm)
                lhs, rhs = split_formula_sides(norm)
                quote = m.group(0).strip()
                extracted.append(
                    ExtractedFormula(
                        raw_latex=content,
                        normalized_latex=norm,
                        lhs=lhs,
                        rhs=rhs,
                        supporting_quote=quote,
                        has_formula=True,
                    )
                )

    # 3. Standalone equation lines (markdown / OCR / plain text math equations)
    for m in EQUATION_LINE_REGEX.finditer(text):
        line = m.group(1).strip()
        norm = normalize_latex_formula(line)
        if norm and len(norm) >= 5 and MATH_SYMBOL_REGEX.search(norm):
            if not any(norm in prev.normalized_latex or prev.normalized_latex in norm for prev in extracted):
                seen_formulas.add(norm)
                lhs, rhs = split_formula_sides(norm)
                extracted.append(
                    ExtractedFormula(
                        raw_latex=line,
                        normalized_latex=norm,
                        lhs=lhs,
                        rhs=rhs,
                        supporting_quote=line,
                        has_formula=True,
                    )
                )

    return extracted


def reconcile_formulas_and_ideas(
    chunk_text: str,
    raw_formulas: List[ExtractedFormula],
    ideas: List[ExtractedIdea],
) -> Tuple[List[ExtractedFormula], List[ExtractedIdea]]:
    """
    Reconciles raw formulas extracted via LaTeX with LLM-extracted ideas.
    - Every genuine conceptual idea extracted by the LLM is retained as a first-class Concept.
      The LLM's conceptual understanding, titles, and explanations must NEVER be deleted or dropped.
    - If an idea formulates or references a mathematical equation, mark idea.has_formula = True,
      associate it with the exact raw formula, and use the LLM's concept title to enrich the formula label.
    - Raw formula preserves exact verbatim LaTeX, while LLM idea preserves conceptual meaning.
    - Promotes substantive unmatched formulas into first-class ExtractedIdea concepts to guarantee
      mathematical formulations and formal equations are never dropped from the knowledge graph.
    """
    if not raw_formulas:
        return raw_formulas, ideas

    reconciled_ideas: List[ExtractedIdea] = []
    matched_formula_indices: Set[int] = set()

    for idea in ideas:
        matched_formula: Optional[ExtractedFormula] = None

        # Check if idea matches or references any raw formula
        for idx, f in enumerate(raw_formulas):
            if f.raw_latex in idea.source_quote or (f.lhs and f.lhs in idea.source_quote):
                matched_formula = f
                matched_formula_indices.add(idx)
            elif f.normalized_latex in normalize_latex_formula(idea.source_quote):
                matched_formula = f
                matched_formula_indices.add(idx)
            elif f.lhs and f.lhs in idea.idea_statement and any(sym in idea.idea_statement for sym in ("=", "≈", "∼", "≤", "≥")):
                matched_formula = f
                matched_formula_indices.add(idx)

            if matched_formula:
                # Enrich formula label from LLM conceptual key entities if formula has no label
                if not f.label and idea.key_entities:
                    for ent in idea.key_entities:
                        if not any(sym in ent for sym in ("=", "\\", "$", "∼")):
                            f.label = ent
                            break
                if not f.label and idea.idea_statement:
                    f.label = idea.idea_statement[:60].strip()
                break

        # Mark has_formula on the idea if it formulated or referenced math
        if matched_formula or idea.idea_type in ("Formula", "Mathematical Formulation") or any(
            f.raw_latex in idea.source_quote or (f.lhs and f.lhs in idea.source_quote) or (f.lhs and f.lhs in idea.idea_statement)
            for f in raw_formulas
        ):
            idea.has_formula = True
            if idea.idea_type not in ("Mathematical Formulation", "Theoretical Claim"):
                idea.idea_type = "Mathematical Formulation"

        # ALWAYS retain the LLM's conceptual idea!
        reconciled_ideas.append(idea)

    # Promote substantive unmatched formulas into first-class ExtractedIdea concepts
    for idx, f in enumerate(raw_formulas):
        if idx in matched_formula_indices:
            continue
        # Check if formula is substantive (has relation operator with LHS and RHS, or long equation)
        is_substantive = bool(f.lhs and f.rhs) or (
            len(f.normalized_latex) >= 6
            and any(c in f.normalized_latex for c in ("=", "\\le", "\\ge", "≤", "≥", "<", ">", "\\approx", "\\sim"))
        )
        if not is_substantive and len(ideas) > 0:
            continue  # Don't promote trivial inline symbols if chunk already has ideas

        formula_label = f.label.strip() if f.label else ""
        if not formula_label:
            if f.lhs and len(f.lhs) <= 40 and not any(sym in f.lhs for sym in ("\\", "$", "\n")):
                formula_label = f"{f.lhs.strip()} Formulation"
            else:
                clean_norm = f.normalized_latex.replace("$", "").strip()
                formula_label = f"Formula: {clean_norm[:35]}"

        synth_idea = ExtractedIdea(
            source_quote=f.supporting_quote or f.raw_latex,
            idea_statement=f"Mathematical formulation for {formula_label}: {f.raw_latex}",
            key_entities=[formula_label, "Mathematical Formulation"],
            idea_type="Mathematical Formulation",
            has_formula=True,
        )
        reconciled_ideas.append(synth_idea)

    return raw_formulas, reconciled_ideas


class SectionExtraction(BaseModel):
    """Extraction output containing all ideas and concepts discovered in a section or chunk."""

    extracted_ideas: List[ExtractedIdea] = Field(
        default_factory=list,
        description="List of core intellectual ideas extracted from the chunk.",
    )
    concepts: List[Concept] = Field(
        default_factory=list,
        description="List of technical ideas, concepts, and architectural patterns discussed.",
    )
    formulas: List[ExtractedFormula] = Field(
        default_factory=list,
        description="List of mathematical formulas extracted from the chunk.",
    )

    @model_validator(mode="before")
    @classmethod
    def sync_ideas_and_concepts(cls, data: Any) -> Any:
        if isinstance(data, dict):
            ideas_raw = data.get("extracted_ideas") or []
            concepts_raw = data.get("concepts") or []

            # Case 1: extracted_ideas provided, concepts empty -> synthesize concepts
            if ideas_raw and not concepts_raw:
                synthesized_concepts = []
                parsed_ideas = []
                for item in ideas_raw:
                    if isinstance(item, dict):
                        idea_obj = ExtractedIdea(**item)
                    elif isinstance(item, ExtractedIdea):
                        idea_obj = item
                    else:
                        continue
                    parsed_ideas.append(idea_obj)
                    synthesized_concepts.append(idea_obj.to_concept())
                data["extracted_ideas"] = parsed_ideas
                data["concepts"] = synthesized_concepts

            # Case 2: concepts provided, extracted_ideas empty -> synthesize extracted_ideas
            elif concepts_raw and not ideas_raw:
                synthesized_ideas = []
                parsed_concepts = []
                for item in concepts_raw:
                    if isinstance(item, dict):
                        concept_obj = Concept(**item)
                    elif isinstance(item, Concept):
                        concept_obj = item
                    else:
                        continue
                    parsed_concepts.append(concept_obj)
                    cat = concept_obj.category.capitalize() if concept_obj.category else "Definition"
                    idea_type = cat if cat in {"Definition", "Argument", "Mechanism", "Example", "Formula"} else "Definition"
                    entities = [concept_obj.name] + [r for r in concept_obj.related_concepts if r != concept_obj.name]
                    synthesized_ideas.append(
                        ExtractedIdea(
                            source_quote=concept_obj.supporting_quote or "",
                            idea_statement=concept_obj.brief_description or concept_obj.detailed_explanation or concept_obj.name,
                            key_entities=entities,
                            idea_type=idea_type,
                            has_formula=concept_obj.has_formula,
                        )
                    )
                data["concepts"] = parsed_concepts
                data["extracted_ideas"] = synthesized_ideas

            # Case 3: Both ideas_raw and concepts_raw empty, but formulas present -> synthesize ideas and concepts
            elif not ideas_raw and not concepts_raw and (data.get("formulas") or []):
                formulas_raw = data.get("formulas") or []
                synthesized_ideas = []
                synthesized_concepts = []
                for f_item in formulas_raw:
                    if isinstance(f_item, dict):
                        f_obj = ExtractedFormula(**f_item)
                    elif isinstance(f_item, ExtractedFormula):
                        f_obj = f_item
                    else:
                        continue
                    label = f_obj.label.strip() if f_obj.label else ""
                    if not label:
                        if f_obj.lhs and len(f_obj.lhs) <= 40 and not any(sym in f_obj.lhs for sym in ("\\", "$", "\n")):
                            label = f"{f_obj.lhs.strip()} Formulation"
                        else:
                            clean_norm = f_obj.normalized_latex.replace("$", "").strip()
                            label = f"Formula: {clean_norm[:35]}"
                    idea = ExtractedIdea(
                        source_quote=f_obj.supporting_quote or f_obj.raw_latex,
                        idea_statement=f"Mathematical formulation for {label}: {f_obj.raw_latex}",
                        key_entities=[label, "Mathematical Formulation"],
                        idea_type="Mathematical Formulation",
                        has_formula=True,
                    )
                    synthesized_ideas.append(idea)
                    synthesized_concepts.append(idea.to_concept())
                data["extracted_ideas"] = synthesized_ideas
                data["concepts"] = synthesized_concepts

            if "formulas" in data and isinstance(data["formulas"], list):
                parsed_f = []
                for item in data["formulas"]:
                    if isinstance(item, dict):
                        parsed_f.append(ExtractedFormula(**item))
                    elif isinstance(item, ExtractedFormula):
                        parsed_f.append(item)
                data["formulas"] = parsed_f
        return data


class ChunkExtraction(BaseModel):
    """Extraction output for an individual chunk within a combined multi-chunk extraction batch."""

    chunk_id: str = Field(
        default="",
        description="The exact identifier or tag of the chunk (matching the <chunk id=\"...\"> tag).",
    )
    extracted_ideas: List[ExtractedIdea] = Field(
        default_factory=list,
        description="List of core intellectual ideas extracted from this specific chunk.",
    )
    concepts: List[Concept] = Field(
        default_factory=list,
        description="List of technical concepts and architectural patterns discussed in this chunk.",
    )
    formulas: List[ExtractedFormula] = Field(
        default_factory=list,
        description="List of mathematical formulas extracted from this specific chunk.",
    )

    @model_validator(mode="before")
    @classmethod
    def sync_chunk_ideas(cls, data: Any) -> Any:
        if isinstance(data, dict):
            if not data.get("chunk_id"):
                for alt_k in ("id", "chunk", "name", "tag"):
                    if data.get(alt_k):
                        data["chunk_id"] = str(data[alt_k])
                        break

            ideas_raw = data.get("extracted_ideas") or []
            concepts_raw = data.get("concepts") or []

            if ideas_raw and not concepts_raw:
                synthesized_concepts = []
                parsed_ideas = []
                for item in ideas_raw:
                    if isinstance(item, dict):
                        idea_obj = ExtractedIdea(**item)
                    elif isinstance(item, ExtractedIdea):
                        idea_obj = item
                    else:
                        continue
                    parsed_ideas.append(idea_obj)
                    synthesized_concepts.append(idea_obj.to_concept())
                data["extracted_ideas"] = parsed_ideas
                data["concepts"] = synthesized_concepts
            elif concepts_raw and not ideas_raw:
                synthesized_ideas = []
                parsed_concepts = []
                for item in concepts_raw:
                    if isinstance(item, dict):
                        concept_obj = Concept(**item)
                    elif isinstance(item, Concept):
                        concept_obj = item
                    else:
                        continue
                    parsed_concepts.append(concept_obj)
                    cat = concept_obj.category.capitalize() if concept_obj.category else "Definition"
                    idea_type = cat if cat in {"Definition", "Argument", "Mechanism", "Example", "Formula"} else "Definition"
                    entities = [concept_obj.name] + [r for r in concept_obj.related_concepts if r != concept_obj.name]
                    synthesized_ideas.append(
                        ExtractedIdea(
                            source_quote=concept_obj.supporting_quote or "",
                            idea_statement=concept_obj.brief_description or concept_obj.detailed_explanation or concept_obj.name,
                            key_entities=entities,
                            idea_type=idea_type,
                            has_formula=concept_obj.has_formula,
                        )
                    )
                data["concepts"] = parsed_concepts
                data["extracted_ideas"] = synthesized_ideas

            if "formulas" in data and isinstance(data["formulas"], list):
                parsed_f = []
                for item in data["formulas"]:
                    if isinstance(item, dict):
                        parsed_f.append(ExtractedFormula(**item))
                    elif isinstance(item, ExtractedFormula):
                        parsed_f.append(item)
                data["formulas"] = parsed_f
        return data


class BatchSectionExtraction(BaseModel):
    """Extraction output containing independent ideas and concepts for multiple combined chunks."""

    chunks: List[ChunkExtraction] = Field(
        default_factory=list,
        description="List of extractions, one entry per combined chunk.",
    )

    @model_validator(mode="before")
    @classmethod
    def sanitize_batch(cls, data: Any) -> Any:
        if isinstance(data, list):
            return {"chunks": data}
        if isinstance(data, dict):
            if "chunks" in data:
                c = data["chunks"]
                if isinstance(c, dict):
                    items = []
                    for cid, cval in c.items():
                        if isinstance(cval, dict):
                            cval_copy = dict(cval)
                            cval_copy.setdefault("chunk_id", str(cid))
                            items.append(cval_copy)
                        elif isinstance(cval, list):
                            items.append({"chunk_id": str(cid), "extracted_ideas": cval})
                    data["chunks"] = items
                return data
            for alt_key in ("chunk_extractions", "extractions", "results", "items"):
                if alt_key in data and isinstance(data[alt_key], list):
                    data["chunks"] = data[alt_key]
                    return data
            if "extracted_ideas" in data or "concepts" in data:
                return {"chunks": [data]}
            # Check if dict keys are chunk_ids
            items = []
            for cid, cval in data.items():
                if isinstance(cval, dict):
                    cval_copy = dict(cval)
                    cval_copy.setdefault("chunk_id", str(cid))
                    items.append(cval_copy)
                elif isinstance(cval, list):
                    items.append({"chunk_id": str(cid), "extracted_ideas": cval})
            if items:
                return {"chunks": items}
        return data


def batch_chunks_by_context_size(
    chunks: Sequence[Any],
    max_context_size: int = 5120,
    max_chunk_batch: Optional[int] = 3,
    tag_overhead: int = 150,
) -> List[List[Any]]:
    """
    Combine consecutive chunks into batches up to max_context_size bytes (default: 5120 bytes / 5KB)
    and up to max_chunk_batch items (default: 3).
    Each chunk's text and its XML chunk tag overhead are accounted for.
    If a single chunk exceeds max_context_size, it is placed in its own single-item batch.
    """
    if max_context_size <= 0:
        return [[c] for c in chunks]

    batches: List[List[Any]] = []
    current_batch: List[Any] = []
    current_size = 0

    def _get_size(c: Any) -> int:
        if isinstance(c, dict):
            raw = c.get("text") or ""
        else:
            raw = getattr(c, "text", "") or ""
        return len(raw.encode("utf-8")) + tag_overhead

    for chk in chunks:
        chk_size = _get_size(chk)
        can_fit_size = (current_size + chk_size <= max_context_size)
        can_fit_count = (max_chunk_batch is None or max_chunk_batch <= 0 or len(current_batch) < max_chunk_batch)

        if not current_batch:
            current_batch.append(chk)
            current_size = chk_size
        elif can_fit_size and can_fit_count:
            current_batch.append(chk)
            current_size += chk_size
        else:
            batches.append(current_batch)
            current_batch = [chk]
            current_size = chk_size

    if current_batch:
        batches.append(current_batch)

    return batches


def validate_concept_chunk_grounding(
    concept_name: str,
    supporting_quote: Optional[str],
    chunk_text: str,
    brief_description: str = "",
) -> Tuple[bool, str]:
    """
    Validate whether an extracted concept has genuine textual grounding in chunk_text.
    Returns (is_grounded, validated_quote).

    - If supporting_quote is present and grounded in chunk_text -> (True, supporting_quote)
    - If supporting_quote is present but ungrounded in chunk_text:
        - If concept_name or brief_description has keyword overlap in chunk -> (True, "") [strip bogus quote]
        - If neither quote nor concept has any presence in chunk -> (False, "") [omit false link]
    - If supporting_quote is not provided:
        - If concept_name or brief_description has keywords in chunk -> (True, "")
        - Otherwise -> (False, "")
    """
    if not chunk_text:
        return False, ""

    _STOP_WORDS = {
        "and", "are", "the", "for", "with", "not", "but", "all", "any", "can",
        "had", "has", "was", "one", "our", "out", "you", "this", "that", "from",
        "they", "been", "have", "were", "into", "more", "some", "such", "than",
    }
    t_clean = BookParser.repair_mojibake(chunk_text).lower()
    t_words = {w for w in re.findall(r"\w{3,}", t_clean) if w not in _STOP_WORDS}

    raw_quote = (supporting_quote or "").strip()
    quote_grounded = False
    valid_quote = ""

    if raw_quote:
        clean_q = BookParser.repair_mojibake(raw_quote)
        q_lower = clean_q.lower()
        # 1. Exact or normalized whitespace substring check
        if " ".join(q_lower.split()) in " ".join(t_clean.split()):
            quote_grounded = True
            valid_quote = clean_q
        else:
            # 2. Significant word overlap check (ignoring tiny punctuation differences)
            q_words = {w for w in re.findall(r"\w{4,}", q_lower) if w not in _STOP_WORDS}
            if not q_words:
                q_words = {w for w in re.findall(r"\w{3,}", q_lower) if w not in _STOP_WORDS}
            if q_words and t_words:
                overlap = len(q_words & t_words) / len(q_words)
                if overlap >= 0.5:
                    quote_grounded = True
                    valid_quote = clean_q

    name_clean = BookParser.repair_mojibake(concept_name).lower()
    name_words = {w for w in re.findall(r"\w{3,}", name_clean) if w not in _STOP_WORDS}
    name_overlap = bool(name_words and (name_words & t_words))

    desc_words = {w for w in re.findall(r"\w{4,}", (brief_description or "").lower()) if w not in _STOP_WORDS}
    desc_overlap = bool(desc_words and len(desc_words & t_words) >= 2)

    if quote_grounded:
        return True, valid_quote

    # Quote was provided but failed grounding
    if raw_quote:
        if name_overlap or desc_overlap:
            # Concept itself appears in chunk, but the quote was inaccurate -> keep concept without quote
            return True, ""
        # Both quote and concept name/description are completely absent
        return False, ""

    # No quote was provided
    if name_overlap or desc_overlap:
        return True, ""

    return False, ""


# ==============================================================================
# KnowledgeExtractor Chain with Multi-Server Failover
# ==============================================================================


class KnowledgeExtractor:
    """Extracts structured metadata and concept graphs using Ollama with multi-server failover."""

    def __init__(
        self,
        base_url: Optional[str] = None,
        model: str = "llama3.1:8b",
        temperature: float = 0.0,
        pool: Optional[OllamaPool] = None,
        cooldown_seconds: int = 600,
        request_timeout: int = 30,
        max_retries: int = 1,
        fallback_model: Optional[str] = None,
        wol_enabled: bool = True,
        wol_wait_seconds: int = 120,
        wol_probe_interval: float = 5.0,
        max_context_size: int = 5120,
        max_chunk_batch: int = 3,
    ):
        self.model_name = model
        self.fallback_model = fallback_model
        self.temperature = temperature
        self.cooldown_seconds = cooldown_seconds
        self.request_timeout = request_timeout
        self.max_retries = max_retries
        self.wol_enabled = wol_enabled
        self.wol_wait_seconds = wol_wait_seconds
        self.wol_probe_interval = wol_probe_interval
        self.max_context_size = max_context_size
        self.max_chunk_batch = max_chunk_batch

        if pool is not None:
            self.pool = pool
        elif base_url:
            urls = [u.strip() for u in base_url.split(",") if u.strip()]
            self.pool = OllamaPool.from_urls(urls, cooldown_seconds=cooldown_seconds)
        else:
            self.pool = OllamaPool.from_urls(["http://localhost:11434"], cooldown_seconds=cooldown_seconds)

        self.base_url = self.pool.primary_server.url

    @classmethod
    def from_settings(
        cls,
        settings: Settings,
        base_url: Optional[str] = None,
        model: Optional[str] = None,
        fallback_model: Optional[str] = None,
        max_context_size: Optional[int] = None,
        max_chunk_batch: Optional[int] = None,
    ) -> "KnowledgeExtractor":
        cooldown = getattr(settings, "resolved_analysis_cooldown_seconds", getattr(settings, "failover_cooldown_seconds", 600))
        if base_url:
            urls = [u.strip() for u in base_url.split(",") if u.strip()]
            pool = OllamaPool.from_urls(
                urls,
                cooldown_seconds=cooldown,
                max_tasks_per_server=1,  # Strictly sequential per Ollama server: no mutual execution
            )
        else:
            servers = getattr(settings, "resolved_llm_servers", None) or settings.resolved_ollama_servers
            pool = OllamaPool(
                servers=servers,
                cooldown_seconds=cooldown,
                max_tasks_per_server=1,  # Strictly sequential per Ollama server: no mutual execution
            )
        eff_max_ctx = getattr(settings, "max_context_size", 5120) if max_context_size is None else max_context_size
        eff_max_batch = getattr(settings, "max_chunk_batch", 3) if max_chunk_batch is None else max_chunk_batch
        return cls(
            pool=pool,
            model=model or settings.llm_model,
            fallback_model=fallback_model or settings.llm_model_fallback,
            cooldown_seconds=cooldown,
            request_timeout=getattr(settings, "request_timeout", 30),
            max_retries=getattr(settings, "max_retries", 1),
            wol_enabled=getattr(settings, "wol_enabled", True),
            wol_wait_seconds=getattr(settings, "wol_wait_seconds", 120),
            wol_probe_interval=getattr(settings, "wol_probe_interval", 5.0),
            max_context_size=eff_max_ctx,
            max_chunk_batch=eff_max_batch,
        )

    def _execute_structured_invoke(
        self,
        schema_cls: Any,
        messages: Any,
        retries: Optional[int] = None,
        quarantine_server: bool = True,
        exclude_urls: Optional[Set[str]] = None,
        model: Optional[str] = None,
        input_chars: int = 0,
        task_type: str = "llm",
    ) -> Any:
        """Execute structured output invocation with multi-server failover, timeout, and retries."""
        timeout_sec = str(self.request_timeout)
        timeout_float = float(self.request_timeout)
        effective_model = model or self.model_name

        if input_chars <= 0 and messages:
            for m in messages:
                content = getattr(m, "content", "")
                if isinstance(content, str):
                    input_chars += len(content)

        prompt_parts = []
        for m in messages:
            c = getattr(m, "content", "")
            if isinstance(c, str):
                prompt_parts.append(c)
            elif isinstance(c, list):
                for sub in c:
                    prompt_parts.append(str(sub.get("text", sub) if isinstance(sub, dict) else sub))
            else:
                prompt_parts.append(str(m))
        full_prompt_text = "\n".join(prompt_parts)

        def _invoke(url: str):
            # 1. Direct curl invocation with format="json" for maximum reliability with Ollama
            if shutil.which("curl"):
                try:
                    msgs_payload = []
                    for m in messages:
                        role = "user"
                        if hasattr(m, "type"):
                            role = "system" if m.type == "system" else "user"
                        content = getattr(m, "content", str(m))
                        msgs_payload.append({"role": role, "content": content})

                    body = {
                        "model": effective_model,
                        "messages": msgs_payload,
                        "format": "json",
                        "options": {
                            "temperature": self.temperature,
                            "top_p": 0.9,
                        },
                        "stream": False,
                    }
                    cmd = [
                        "curl",
                        "-s",
                        "--connect-timeout",
                        "3",
                        "--max-time",
                        timeout_sec,
                        "-X",
                        "POST",
                        "-H",
                        "Content-Type: application/json",
                        "-d",
                        json.dumps(body),
                        f"{url.rstrip('/')}/api/chat",
                    ]
                    res = subprocess.run(cmd, capture_output=True, text=True, check=False)
                    if res.returncode == 0 and res.stdout.strip():
                        data = json.loads(res.stdout)
                        content = data.get("message", {}).get("content", "")
                        try:
                            from bookeeper.processing.token_tracker import TokenTracker
                            TokenTracker.get_instance().record_response_data(
                                data,
                                prompt_text=full_prompt_text,
                                completion_text=content,
                                task_type="llm",
                            )
                        except Exception as track_err:
                            logger.debug(f"Failed to record token usage: {track_err}")
                        if "<think>" in content and "</think>" in content:
                            content = content.split("</think>")[-1].strip()
                        if content and content.strip():
                            content = content.strip()
                            if content.startswith("```json"):
                                content = content[7:]
                            elif content.startswith("```"):
                                content = content[3:]
                            if content.endswith("```"):
                                content = content[:-3]
                            content = content.strip()
                            if hasattr(schema_cls, "model_validate_json"):
                                return schema_cls.model_validate_json(content)
                            return json.loads(content)
                    elif res.returncode == 7:
                        raise ConnectionRefusedError(f"Ollama server {url} unreachable / connection refused (curl code 7)")
                    elif res.returncode == 28:
                        raise TimeoutError(f"Ollama request to {url}/api/chat timed out after {timeout_sec}s")
                    elif res.returncode != 0:
                        logger.debug(f"Direct curl format=json returned non-zero code {res.returncode}")
                except (TimeoutError, ConnectionError):
                    raise
                except Exception as curl_err:
                    logger.debug(f"Direct curl format=json invoke failed: {curl_err}")

            # 2. Fallback to ChatOllama structured output chain with strict timeout
            try:
                # Clean sampling parameters: pass only valid parameters (no mirostat, mirostat_eta, mirostat_tau, or tfs_z)
                llm = ChatOllama(
                    base_url=url,
                    model=effective_model,
                    temperature=self.temperature,
                    top_p=0.9,
                    sync_client_kwargs={"timeout": timeout_float},
                    client_kwargs={"timeout": timeout_float},
                )
                chain = llm.with_structured_output(schema_cls)
                inv_res = chain.invoke(messages)
                try:
                    from bookeeper.processing.token_tracker import TokenTracker, estimate_token_count
                    p_toks = estimate_token_count(full_prompt_text)
                    o_toks = estimate_token_count(str(inv_res))
                    TokenTracker.get_instance().record_llm_tokens(input_tokens=p_toks, output_tokens=o_toks)
                except Exception as track_err:
                    logger.debug(f"Failed to record fallback token usage: {track_err}")
                return inv_res
            except Exception as exc:
                raise exc

        return self.pool.execute_with_failover(
            _invoke,
            retries=self.max_retries if retries is None else retries,
            max_task_duration=max(60.0, 2.5 * self.request_timeout),
            model_name=effective_model,
            quarantine_server=quarantine_server,
            exclude_urls=exclude_urls,
            capability="llm",
            input_chars=input_chars,
            task_type=task_type,
        )

    def clean_metadata(
        self,
        raw_title: str,
        raw_authors: List[str],
        raw_comments: Optional[str] = None,
        content_sample: Optional[str] = None,
        file_hint: Optional[str] = None,
        retries: Optional[int] = None,
        model: Optional[str] = None,
    ) -> BookMetadata:
        """Clean and normalize title, authors, and summary blurb using content sampling."""
        from bookeeper.calibre.parser import BookParser

        raw_title = BookParser.repair_mojibake(raw_title)
        raw_authors = [BookParser.repair_mojibake(a) for a in raw_authors]
        if raw_comments:
            raw_comments = BookParser.repair_mojibake(raw_comments)
        if content_sample:
            content_sample = BookParser.repair_mojibake(content_sample)

        authors_str = ", ".join(raw_authors) if raw_authors else "Unknown"
        comments_str = (raw_comments or "").strip()

        system_instruction = (
            "You are an expert multilingual library cataloguer and metadata specialist.\n"
            "Your task is to identify the canonical book title, author(s), and generate an accurate "
            "executive summary blurb (2-4 sentences) describing the book's premise, themes, and content.\n"
            "Important guidelines:\n"
            "1. Multilingual Support: Preserve the book's canonical language and script (e.g. Russian Cyrillic, "
            "English, French, etc.). If the book is in Russian, produce the title, author, and summary in Russian.\n"
            "2. Identify Corrupted or Missing Metadata: If the raw Calibre title/author has encoding errors "
            "(e.g., mojibake like 'Ñóïåðìåí Ïðèêëþ÷åíèÿ' which is Windows-1251 decoded as Latin-1 for 'Супермен Приключения', "
            "or ' 2033'), missing data, generic placeholders ('Unknown'), or is just a filename ('zubkov'), identify "
            "the actual canonical title and author from the content excerpt, RTF/OPF headers, and file hints.\n"
            "   - ALWAYS output clean, natural UTF-8 text in the book's proper script. NEVER echo garbled mojibake characters like 'Ñóïåðìåí' or 'Ð.Ð.Ð.'.\n"
            "   - E.g. ' 2033' / '2033 - .rtf' is 'Метро 2033' by 'Дмитрий Глуховский'.\n"
            "   - E.g. 'zubkov' / 'zubkov.djvu' in Russian computer science refers to 'Сергей Зубков' (С. В. Зубков), "
            "author of 'Ассемблер. Для DOS, Windows и UNIX'.\n"
            "3. Comprehensive Summary: ALWAYS generate a rich, informative 2-4 sentence summary blurb describing "
            "what the book is about based on the excerpt, premise, or subject. NEVER output 'No summary available', "
            "'None', or placeholder text."
        )

        user_content = (
            f"Please clean and standardize this book's metadata:\n\n"
            f"Raw Calibre Title: {raw_title}\n"
            f"Raw Calibre Authors: {authors_str}\n"
            f"Existing Comments / Blurb: {comments_str or '(None)'}\n"
        )
        if file_hint:
            user_content += f"File / Format Hint: {file_hint}\n"
        if content_sample:
            user_content += f"\nBook Content Excerpt (Title page / Preface / Beginning):\n\"\"\"\n{content_sample[:3500]}\n\"\"\"\n"

        messages = [
            SystemMessage(content=system_instruction),
            HumanMessage(content=user_content),
        ]

        meta_chars = len(raw_title) + sum(len(a) for a in raw_authors) + len(raw_comments or "") + len(content_sample or "")

        try:
            result = self._execute_structured_invoke(
                BookMetadata,
                messages,
                retries=retries,
                model=model,
                input_chars=meta_chars,
                task_type="llm",
            )
            if isinstance(result, BookMetadata):
                result.title = BookParser.repair_mojibake(result.title)
                result.author = BookParser.repair_mojibake(result.author)
                result.summary = BookParser.repair_mojibake(result.summary)
                return result
            data = dict(result)
            return BookMetadata(
                title=BookParser.repair_mojibake(data.get("title", raw_title)),
                author=BookParser.repair_mojibake(data.get("author", authors_str)),
                summary=BookParser.repair_mojibake(data.get("summary", "No summary available.")),
            )
        except Exception as e:
            logger.warning(f"Metadata cleaning LLM call failed across all pool servers: {e}. Falling back to raw.")
            return BookMetadata(
                title=raw_title,
                author=authors_str,
                summary=comments_str[:300] if comments_str else "No summary available.",
            )

    def extract_section(
        self,
        text: str,
        book_title: str,
        section_title: str,
        subtitle: Optional[str] = None,
        parent_context: Optional[str] = None,
        retries: Optional[int] = None,
        quarantine_server: bool = True,
        exclude_urls: Optional[Set[str]] = None,
        raise_on_error: bool = False,
        model: Optional[str] = None,
    ) -> SectionExtraction:
        """
        Extract canonical domain ideas, principles, patterns, and their relationships
        from a section or chunk with multi-server failover.
        Supports parent macro context for hierarchical RAG understanding.
        """
        system_text = (
            "You are an expert knowledge-graph extractor. Your goal is to extract core intellectual ideas, mathematical formulations, empirical findings, and architectural mechanisms from the provided book chunk.\n\n"
            "Rules:\n"
            "1. Every idea must be anchored by an exact verbatim quote from the text.\n"
            '2. The "idea_statement" must be a concise, self-contained factual claim (1–2 sentences).\n'
            "3. Do not invent details not present in the chunk.\n"
            "4. Extract only distinct, meaningful concepts (usually 1 to 4 per chunk). Specifically capture:\n"
            "   - Mathematical Formulations & Objectives: Key loss equations, optimization objectives, error bounds, and theorem/lemma statements.\n"
            "   - Empirical Findings & Failure Modes: Discovered empirical trade-offs, scaling limits, ablation drops, and behavioral vulnerabilities.\n"
            "   - Protocols & Algorithmic Loops: Architectural workflows, decision gates, formal verification rules, and agent interaction patterns.\n"
            "   - Conceptual Definitions & Core Arguments.\n"
            "   If the chunk contains no standalone ideas or is generic boilerplate, output an empty list.\n"
            "5. Specificity in Key Entities (CRITICAL: Primary Entity is the Concept Title):\n"
            "   - The FIRST entry in 'key_entities' becomes the canonical title of the concept in the Knowledge Graph.\n"
            "   - NEVER use the paper's overarching framework, library, tool, system, or book name (e.g., 'Grasp', 'EvoAlloc', 'LLaVA', 'Model', 'Gradient', 'Paper') as the primary key entity.\n"
            "   - Instead, name the specific mechanism, mathematical formulation, rule, or empirical finding (e.g. 'Mutagenic LUMO-Threshold Rule', 'Reasoning-First XML Trace Protocol', 'Answer-First JSON Degradation').\n"
            "6. Dense Math & Symbolic Code Chunks:\n"
            "   - When a chunk contains mathematical equations, proofs, bounds (e.g. spectral radius bound rho_P < 1), or formal code (Lean 4, Prolog logic predicates), extract the specific mathematical bound, loss formulation, or logical deduction as a named concept rather than skipping it or outputting empty.\n"
            "7. Return strictly valid JSON matching the schema below.\n"
            "Output Format:\n"
            "{\n"
            '  "extracted_ideas": [\n'
            "    {\n"
            '      "source_quote": "Exact verbatim sentence from the chunk",\n'
            '      "idea_statement": "Concise summary of the concept or argument",\n'
            '      "key_entities": ["Specific Concept Title", "Secondary Topic"],\n'
            '      "idea_type": "Definition" | "Argument" | "Mechanism" | "Example" | "Mathematical Formulation" | "Theoretical Claim" | "Empirical Finding" | "Protocol"\n'
            "    }\n"
            "  ]\n"
            "}\n\n"
            "Examples:\n"
            "Example 1 (Mechanism):\n"
            "Chunk:\n"
            '"Systems theory demonstrates that complex networks exhibit emergent properties. In ant colonies, no single ant understands the global architecture of the nest, yet through simple local pheromone interactions, highly coordinated structures arise spontaneously."\n\n'
            "Response:\n"
            "{\n"
            '  "extracted_ideas": [\n'
            "    {\n"
            '      "source_quote": "In ant colonies, no single ant understands the global architecture of the nest, yet through simple local pheromone interactions, highly coordinated structures arise spontaneously.",\n'
            '      "idea_statement": "Complex coordination emerges from local rules without central leadership.",\n'
            '      "key_entities": ["Systems Theory", "Emergence", "Self-organization"],\n'
            '      "idea_type": "Mechanism"\n'
            "    }\n"
            "  ]\n"
            "}\n\n"
            "Example 2 (Mathematical Formulation):\n"
            "Chunk:\n"
            '"We define the option-conditioned multi-step temporal-difference loss over interval k as LQh,phi = E_{(s_t, s_{t+k}, g)} [(Q_h(s_t, g, phi(s_t, s_{t+k})) - R_h(s_{t:t+k}, g) - gamma V_h(s_{t+k}, g))^2]."\n\n'
            "Response:\n"
            "{\n"
            '  "extracted_ideas": [\n'
            "    {\n"
            '      "source_quote": "We define the option-conditioned multi-step temporal-difference loss over interval k as LQh,phi = E_{(s_t, s_{t+k}, g)} [(Q_h(s_t, g, phi(s_t, s_{t+k})) - R_h(s_{t:t+k}, g) - gamma V_h(s_{t+k}, g))^2].",\n'
            '      "idea_statement": "Option-conditioned multi-step TD objective measures squared Bellman error over k-step transition subgoals.",\n'
            '      "key_entities": ["Hierarchical TD Loss", "Option-Conditioned Q-Learning", "Subgoal Transition Interval"],\n'
            '      "idea_type": "Mathematical Formulation"\n'
            "    }\n"
            "  ]\n"
            "}\n\n"
            "Example 3 (Empirical Finding):\n"
            "Chunk:\n"
            '"Under constrained JSON formats, placing the answer field before the rationale forces the model to prematurely commit to a single incorrect path, causing accuracy to collapse from 82% to 41%."\n\n'
            "Response:\n"
            "{\n"
            '  "extracted_ideas": [\n'
            "    {\n"
            '      "source_quote": "Under constrained JSON formats, placing the answer field before the rationale forces the model to prematurely commit to a single incorrect path, causing accuracy to collapse from 82% to 41%.",\n'
            '      "idea_statement": "Answer-first structured JSON schemas induce premature commitment, halving reasoning accuracy compared to reasoning-first generation.",\n'
            '      "key_entities": ["Answer-First JSON Degradation", "Reasoning-First XML Trace Protocol", "Format Sensitivity"],\n'
            '      "idea_type": "Empirical Finding"\n'
            "    }\n"
            "  ]\n"
            "}"
        )

        book_title = BookParser.repair_mojibake(book_title)
        section_title = BookParser.repair_mojibake(section_title)
        subtitle = BookParser.repair_mojibake(subtitle) if subtitle else ""
        text = BookParser.repair_mojibake(text)
        if parent_context:
            parent_context = BookParser.repair_mojibake(parent_context)

        loc = ""
        if book_title or section_title:
            loc = f"Context: Book: '{book_title}' | Section: '{section_title}'"
            if subtitle and subtitle != section_title:
                loc += f" | Subsection: '{subtitle}'"
            loc += "\n\n"

        human_text = (
            f"{loc}"
            "Now process the following text:\n\n"
            "Chunk:\n"
            '"""\n'
            f"{text[:4000]}\n"
            '"""'
        )
        if parent_context and parent_context != text:
            human_text += (
                f"\n\nBackground Context (Surrounding passage provided ONLY for narrative orientation - DO NOT extract concepts or quotes from here):\n"
                f'"""\n{parent_context[:2000]}\n"""'
            )

        messages = [
            SystemMessage(content=system_text),
            HumanMessage(content=human_text),
        ]

        chunk_chars = len(text)

        # Check if chunk contains mathematical formulas and pre-extract raw formulas
        chunk_has_formula = has_mathematical_formula(text)
        raw_formulas: List[ExtractedFormula] = []
        if chunk_has_formula:
            raw_formulas = extract_raw_formulas_from_text(text)

        try:
            result = self._execute_structured_invoke(
                SectionExtraction,
                messages,
                retries=retries,
                quarantine_server=quarantine_server,
                exclude_urls=exclude_urls,
                model=model,
                input_chars=chunk_chars,
                task_type="llm",
            )
            if isinstance(result, SectionExtraction):
                extraction_obj = result
            else:
                extraction_obj = SectionExtraction(**dict(result))

            # Validate quote grounding for extracted ideas
            validated_ideas = []
            for idea in extraction_obj.extracted_ideas:
                is_grounded, val_quote = validate_concept_chunk_grounding(
                    concept_name=idea.idea_statement,
                    supporting_quote=idea.source_quote,
                    chunk_text=text,
                    brief_description=idea.idea_statement,
                )
                if is_grounded:
                    if val_quote:
                        idea.source_quote = val_quote
                    validated_ideas.append(idea)

            if raw_formulas:
                raw_formulas, validated_ideas = reconcile_formulas_and_ideas(
                    text, raw_formulas, validated_ideas
                )

            extraction_obj.extracted_ideas = validated_ideas
            extraction_obj.concepts = [i.to_concept() for i in validated_ideas]
            extraction_obj.formulas = raw_formulas
            return extraction_obj
        except Exception as e:
            if raise_on_error:
                raise
            logger.warning(f"Section extraction LLM call failed across all pool servers for '{section_title}': {e}")
            return SectionExtraction(concepts=[], extracted_ideas=[], formulas=raw_formulas)

    def extract_ideas(
        self,
        text: str,
        book_title: str = "",
        section_title: str = "",
        subtitle: Optional[str] = None,
        parent_context: Optional[str] = None,
        retries: Optional[int] = None,
        quarantine_server: bool = True,
        exclude_urls: Optional[Set[str]] = None,
        raise_on_error: bool = False,
        model: Optional[str] = None,
    ) -> List[ExtractedIdea]:
        """
        Extract core intellectual ideas from chunk text using exact knowledge-graph prompt.
        Returns list of ExtractedIdea instances.
        """
        extraction = self.extract_section(
            text=text,
            book_title=book_title,
            section_title=section_title,
            subtitle=subtitle,
            parent_context=parent_context,
            retries=retries,
            quarantine_server=quarantine_server,
            exclude_urls=exclude_urls,
            raise_on_error=raise_on_error,
            model=model,
        )
        return extraction.extracted_ideas

    def extract_ideas(
        self,
        text: str,
        book_title: str = "Unknown",
        section_title: str = "Unknown",
        parent_context: Optional[str] = None,
        model: Optional[str] = None,
    ) -> List[Concept]:
        """Convenience method to extract canonical concepts/ideas from text using the Oracle model."""
        result = self.extract_section(
            text=text,
            book_title=book_title,
            section_title=section_title,
            parent_context=parent_context,
            model=model,
        )
        return result.concepts

    extract_concepts_from_section = extract_section

    def batch_chunks(
        self,
        chunks: Sequence[Any],
        max_context_size: Optional[int] = None,
        max_chunk_batch: Optional[int] = None,
    ) -> List[List[Any]]:
        """
        Combine chunks into batches up to maximum context size (default: self.max_context_size / 5KB)
        and maximum chunk count (default: self.max_chunk_batch / 3).
        """
        size_limit = self.max_context_size if max_context_size is None else max_context_size
        count_limit = self.max_chunk_batch if max_chunk_batch is None else max_chunk_batch
        return batch_chunks_by_context_size(chunks, max_context_size=size_limit, max_chunk_batch=count_limit)

    def extract_chunks(
        self,
        chunks: List[Any],
        book_title: str = "",
        model: Optional[str] = None,
        retries: Optional[int] = None,
        quarantine_server: bool = True,
        exclude_urls: Optional[Set[str]] = None,
        raise_on_error: bool = False,
    ) -> Dict[str, SectionExtraction]:
        """
        Extract core intellectual ideas simultaneously from multiple chunks combined into
        a single LLM prompt with XML tags, returning independent SectionExtraction outputs for each chunk.
        """
        if not chunks:
            return {}

        def _get_id(c: Any) -> str:
            if isinstance(c, dict):
                return str(c.get("chunk_id") or c.get("id") or "")
            return str(getattr(c, "chunk_id", getattr(c, "id", "")))

        def _get_text(c: Any) -> str:
            if isinstance(c, dict):
                return str(c.get("text") or "")
            return str(getattr(c, "text", "") or "")

        def _get_section(c: Any) -> str:
            if isinstance(c, dict):
                return str(c.get("section_title") or c.get("section") or "")
            return str(getattr(c, "section_title", getattr(c, "section", "")) or "")

        def _get_subtitle(c: Any) -> str:
            if isinstance(c, dict):
                return str(c.get("subtitle") or "")
            return str(getattr(c, "subtitle", "") or "")

        def _get_idx(c: Any) -> Any:
            if isinstance(c, dict):
                return c.get("chunk_idx") or c.get("index") or 0
            return getattr(c, "chunk_idx", getattr(c, "index", 0))

        # Single chunk fast-path (preserves 100% exact legacy behavior)
        if len(chunks) == 1:
            c = chunks[0]
            cid = _get_id(c)
            ctext = _get_text(c)
            csec = _get_section(c)
            csub = _get_subtitle(c)
            b_title = book_title or (c.get("book_title") if isinstance(c, dict) else getattr(c, "book_title", ""))
            parent_ctx = c.get("parent_text") if isinstance(c, dict) else getattr(c, "parent_text", None)
            res = self.extract_section(
                text=ctext,
                book_title=b_title,
                section_title=csec,
                subtitle=csub,
                parent_context=parent_ctx,
                retries=retries,
                quarantine_server=quarantine_server,
                exclude_urls=exclude_urls,
                raise_on_error=raise_on_error,
                model=model,
            )
            return {cid: res}

        # Multi-chunk simultaneous extraction with appropriate XML chunk tags
        system_text = (
            "You are an expert knowledge-graph extractor. Your goal is to extract core intellectual ideas, mathematical formulations, empirical findings, and architectural mechanisms from the provided book chunks.\n"
            'Multiple chunks are provided simultaneously, each enclosed within <chunk id="..." section="..." index="..."> ... </chunk> tags.\n\n'
            "Rules:\n"
            "1. Process each chunk independently and return separate extracted ideas for each chunk id.\n"
            "2. Every idea must be anchored by an exact verbatim quote from that specific chunk's text.\n"
            '3. The "idea_statement" must be a concise, self-contained factual claim (1–2 sentences).\n'
            "4. Do not invent details not present in the chunk. Do not mix quotes or concepts across different chunks.\n"
            "5. Extract only distinct, meaningful concepts (usually 1 to 4 per chunk). Specifically capture:\n"
            "   - Mathematical Formulations & Objectives: Key loss equations, optimization objectives, error bounds, and theorem/lemma statements.\n"
            "   - Empirical Findings & Ablations: Discovered empirical trade-offs, scaling limits, ablation drops, and behavioral vulnerabilities.\n"
            "   - Protocols & Algorithmic Loops: Architectural workflows, decision gates, formal verification rules, and agent interaction patterns.\n"
            "   - Conceptual Definitions & Core Arguments.\n"
            "   If a chunk contains no standalone ideas or is generic boilerplate, output an empty list of extracted ideas for that chunk id.\n"
            "6. Specificity in Key Entities (CRITICAL: Primary Entity is the Concept Title):\n"
            "   - The FIRST entry in 'key_entities' becomes the canonical title of the concept in the Knowledge Graph.\n"
            "   - NEVER use the paper's overarching framework, library, tool, system, or book name (e.g., 'Grasp', 'EvoAlloc', 'LLaVA', 'Model', 'Gradient', 'Paper') as the primary key entity.\n"
            "   - Instead, name the specific mechanism, mathematical formulation, rule, or empirical finding (e.g. 'Mutagenic LUMO-Threshold Rule', 'Reasoning-First XML Trace Protocol', 'Answer-First JSON Degradation').\n"
            "7. Dense Math & Symbolic Code Chunks:\n"
            "   - When a chunk contains mathematical equations, proofs, bounds (e.g. spectral radius bound rho_P < 1), or formal code (Lean 4, Prolog logic predicates), extract the specific mathematical bound, loss formulation, or logical deduction as a named concept rather than skipping it or outputting empty.\n"
            "8. Return strictly valid JSON matching the schema below.\n\n"
            "Output Format:\n"
            "{\n"
            '  "chunks": [\n'
            "    {\n"
            '      "chunk_id": "Exact chunk ID matching the <chunk id=\\"...\\"> tag",\n'
            '      "extracted_ideas": [\n'
            "        {\n"
            '          "source_quote": "Exact verbatim sentence from this specific chunk",\n'
            '          "idea_statement": "Concise summary of the concept or argument",\n'
            '          "key_entities": ["Specific Concept Title", "Secondary Topic"],\n'
            '          "idea_type": "Definition" | "Argument" | "Mechanism" | "Example" | "Mathematical Formulation" | "Theoretical Claim" | "Empirical Finding" | "Protocol"\n'
            "        }\n"
            "      ]\n"
            "    }\n"
            "  ]\n"
            "}\n\n"
            "Example:\n"
            "Input:\n"
            '<chunk id="c1" section="Chapter 1" index="1">\n'
            "Systems theory demonstrates that complex networks exhibit emergent properties. In ant colonies, no single ant understands the global architecture of the nest, yet through simple local pheromone interactions, highly coordinated structures arise spontaneously.\n"
            "</chunk>\n\n"
            '<chunk id="c2" section="Chapter 1" index="2">\n'
            "In contrast, hierarchical top-down control structures suffer from single points of failure. When a centralized coordinator crashes, the entire distributed cluster halts unless automated leader election protocols intervene.\n"
            "</chunk>\n\n"
            "Response:\n"
            "{\n"
            '  "chunks": [\n'
            "    {\n"
            '      "chunk_id": "c1",\n'
            '      "extracted_ideas": [\n'
            "        {\n"
            '          "source_quote": "In ant colonies, no single ant understands the global architecture of the nest, yet through simple local pheromone interactions, highly coordinated structures arise spontaneously.",\n'
            '          "idea_statement": "Complex coordination emerges from local rules without central leadership.",\n'
            '          "key_entities": ["Systems Theory", "Emergence", "Self-organization"],\n'
            '          "idea_type": "Mechanism"\n'
            "        }\n"
            "      ]\n"
            "    },\n"
            "    {\n"
            '      "chunk_id": "c2",\n'
            '      "extracted_ideas": [\n'
            "        {\n"
            '          "source_quote": "When a centralized coordinator crashes, the entire distributed cluster halts unless automated leader election protocols intervene.",\n'
            '          "idea_statement": "Centralized systems risk catastrophic failure upon coordinator crash without automated leader election.",\n'
            '          "key_entities": ["Centralized Systems", "Leader Election", "Fault Tolerance"],\n'
            '          "idea_type": "Argument"\n'
            "        }\n"
            "      ]\n"
            "    }\n"
            "  ]\n"
            "}"
        )

        b_title = BookParser.repair_mojibake(book_title)
        loc = f"Context: Book: '{b_title}'\n\n" if b_title else ""

        chunk_blocks = []
        total_input_chars = len(loc)
        chunk_raw_formulas: Dict[str, List[ExtractedFormula]] = {}
        for c in chunks:
            cid = _get_id(c)
            csec = BookParser.repair_mojibake(_get_section(c))
            csub = BookParser.repair_mojibake(_get_subtitle(c))
            cidx = _get_idx(c)
            ctext = BookParser.repair_mojibake(_get_text(c))
            c_has_formula = bool(
                getattr(c, "has_formula", False)
                or (isinstance(c, dict) and c.get("has_formula", False))
                or has_mathematical_formula(ctext)
            )
            if c_has_formula:
                chunk_raw_formulas[cid] = extract_raw_formulas_from_text(ctext)
            else:
                chunk_raw_formulas[cid] = []
            tag_attrs = f'id="{cid}"'
            if csec:
                tag_attrs += f' section="{csec}"'
            if csub and csub != csec:
                tag_attrs += f' subtitle="{csub}"'
            if cidx:
                tag_attrs += f' index="{cidx}"'
            block = f"<chunk {tag_attrs}>\n{ctext}\n</chunk>"
            chunk_blocks.append(block)
            total_input_chars += len(block)

        human_text = (
            f"{loc}"
            "Now process the following chunks independently and provide separate output for each chunk:\n\n"
            + "\n\n".join(chunk_blocks)
        )

        messages = [
            SystemMessage(content=system_text),
            HumanMessage(content=human_text),
        ]

        try:
            raw_result = self._execute_structured_invoke(
                BatchSectionExtraction,
                messages,
                retries=retries,
                quarantine_server=quarantine_server,
                exclude_urls=exclude_urls,
                model=model,
                input_chars=total_input_chars,
                task_type="llm",
            )
            if isinstance(raw_result, BatchSectionExtraction):
                batch_obj = raw_result
            elif isinstance(raw_result, dict):
                batch_obj = BatchSectionExtraction.model_validate(raw_result)
            else:
                batch_obj = BatchSectionExtraction.model_validate(dict(raw_result))

            # Build lookup of chunk extractions returned by model
            model_extractions: Dict[str, ChunkExtraction] = {}
            for item in batch_obj.chunks:
                if item.chunk_id:
                    model_extractions[str(item.chunk_id).strip()] = item

            results: Dict[str, SectionExtraction] = {}
            for idx, c in enumerate(chunks):
                cid = _get_id(c)
                ctext = _get_text(c)
                matching_item = model_extractions.get(cid)
                if not matching_item:
                    # Try partial or index-based matching
                    for m_id, m_val in model_extractions.items():
                        if cid in m_id or m_id in cid or m_id == f"chunk_{idx+1}" or m_id == str(idx+1):
                            matching_item = m_val
                            break

                if matching_item:
                    validated_ideas = []
                    for idea in matching_item.extracted_ideas:
                        is_grounded, val_quote = validate_concept_chunk_grounding(
                            concept_name=idea.idea_statement,
                            supporting_quote=idea.source_quote,
                            chunk_text=ctext,
                            brief_description=idea.idea_statement,
                        )
                        if is_grounded:
                            if val_quote:
                                idea.source_quote = val_quote
                            validated_ideas.append(idea)

                    raw_formulas = chunk_raw_formulas.get(cid, [])
                    if raw_formulas:
                        raw_formulas, validated_ideas = reconcile_formulas_and_ideas(
                            ctext, raw_formulas, validated_ideas
                        )

                    results[cid] = SectionExtraction(
                        extracted_ideas=validated_ideas,
                        concepts=[i.to_concept() for i in validated_ideas],
                        formulas=raw_formulas,
                    )
                else:
                    # Missing from batch output: fallback to single extraction
                    try:
                        csec = _get_section(c)
                        csub = _get_subtitle(c)
                        parent_ctx = c.get("parent_text") if isinstance(c, dict) else getattr(c, "parent_text", None)
                        single_res = self.extract_section(
                            text=ctext,
                            book_title=book_title,
                            section_title=csec,
                            subtitle=csub,
                            parent_context=parent_ctx,
                            retries=retries,
                            quarantine_server=quarantine_server,
                            exclude_urls=exclude_urls,
                            raise_on_error=False,
                            model=model,
                        )
                        results[cid] = single_res
                    except Exception:
                        results[cid] = SectionExtraction(
                            concepts=[],
                            extracted_ideas=[],
                            formulas=chunk_raw_formulas.get(cid, []),
                        )

            return results

        except Exception as e:
            if raise_on_error:
                raise
            logger.warning(f"Batch chunk extraction LLM call failed across pool servers: {e}")
            return {
                _get_id(c): SectionExtraction(
                    concepts=[],
                    extracted_ideas=[],
                    formulas=chunk_raw_formulas.get(_get_id(c), []),
                )
                for c in chunks
            }

    def warmup_and_check_device(
        self,
        timeout: int = 15,
        on_status_callback: Optional[Callable[[str], None]] = None,
    ) -> Dict[str, Any]:
        """
        Trigger warmup requests and query /api/ps across all configured Ollama servers in the pool.
        Returns metrics for each server, identifying GPU (VRAM) vs CPU execution and failover priority.
        If any server is unreachable and has a configured MAC address, Wake-on-LAN packet is broadcast,
        followed by a wait period (up to wol_wait_seconds) to allow the machine to boot and retry warmup.
        """
        def _fetch_server_json(base_url: str, endpoint: str, post_body: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
            url = f"{base_url.rstrip('/')}{endpoint}"
            try:
                req = urllib.request.Request(
                    url,
                    data=json.dumps(post_body).encode("utf-8") if post_body else None,
                    headers={"Content-Type": "application/json"} if post_body else {},
                )
                with urllib.request.urlopen(req, timeout=timeout) as resp:
                    return json.loads(resp.read().decode("utf-8"))
            except Exception:
                if shutil.which("curl"):
                    cmd = ["curl", "-s", "--max-time", str(timeout)]
                    if post_body:
                        cmd.extend(["-X", "POST", "-H", "Content-Type: application/json", "-d", json.dumps(post_body)])
                    cmd.append(url)
                    res = subprocess.run(cmd, capture_output=True, text=True, check=False)
                    if res.returncode == 0 and res.stdout.strip():
                        return json.loads(res.stdout)
                raise

        def _inspect_node(node) -> Dict[str, Any]:
            target_endpoint = getattr(node, "target_url", node.url)
            # 1. Warmup ping
            try:
                _fetch_server_json(
                    target_endpoint,
                    "/api/generate",
                    {"model": self.model_name, "prompt": "warmup ping", "options": {"num_predict": 1}, "stream": False},
                )
            except Exception as e:
                logger.debug(f"Warmup ping error for {node.url}: {e}")

            # 2. Query /api/ps
            try:
                ps_data = _fetch_server_json(target_endpoint, "/api/ps")
                models = ps_data.get("models", [])

                matched = None
                target_norm = self.model_name.lower()
                for m in models:
                    m_name = (m.get("name") or m.get("model") or "").lower()
                    if m_name == target_norm or m_name.startswith(target_norm) or target_norm.startswith(m_name.split(":")[0]):
                        matched = m
                        break

                if not matched and models:
                    matched = models[0]

                if matched:
                    size = matched.get("size", 0)
                    size_vram = matched.get("size_vram", 0)
                    runner = matched.get("runner", "llamacpp")
                    model_active = matched.get("name") or self.model_name
                    node.mark_success()

                    if size_vram == 0:
                        return {
                            "url": node.url,
                            "priority": node.priority,
                            "name": node.label,
                            "status": "ok",
                            "is_gpu": False,
                            "device": "CPU",
                            "size": size,
                            "size_vram": 0,
                            "vram_pct": 0.0,
                            "runner": runner,
                            "model": model_active,
                            "warning": f"Running on CPU (0% VRAM offload).",
                        }
                    elif size_vram >= size:
                        return {
                            "url": node.url,
                            "priority": node.priority,
                            "name": node.label,
                            "status": "ok",
                            "is_gpu": True,
                            "device": "GPU (100% VRAM offload)",
                            "size": size,
                            "size_vram": size_vram,
                            "vram_pct": 100.0,
                            "runner": runner,
                            "model": model_active,
                            "warning": None,
                        }
                    else:
                        pct = round((size_vram / size) * 100, 1)
                        return {
                            "url": node.url,
                            "priority": node.priority,
                            "name": node.label,
                            "status": "ok",
                            "is_gpu": True,
                            "device": f"Partial GPU ({pct}% VRAM offload)",
                            "size": size,
                            "size_vram": size_vram,
                            "vram_pct": pct,
                            "runner": runner,
                            "model": model_active,
                            "warning": f"Partially offloaded ({pct}% VRAM, remainder on CPU).",
                        }

                node.mark_success()
                return {
                    "url": node.url,
                    "priority": node.priority,
                    "name": node.label,
                    "status": "idle",
                    "is_gpu": False,
                    "device": "Idle/Ready",
                    "size": 0,
                    "size_vram": 0,
                    "vram_pct": 0.0,
                    "runner": "unknown",
                    "model": self.model_name,
                    "warning": None,
                }
            except Exception as e:
                node.mark_failure(str(e))
                return {
                    "url": node.url,
                    "priority": node.priority,
                    "name": node.label,
                    "status": "unreachable",
                    "is_gpu": False,
                    "device": "Unreachable",
                    "size": 0,
                    "size_vram": 0,
                    "vram_pct": 0.0,
                    "runner": "unknown",
                    "model": self.model_name,
                    "warning": f"Unreachable: {e}",
                    "error": str(e),
                }

        server_reports = [_inspect_node(n) for n in self.pool.nodes]

        # Wake-on-LAN: If any server is unreachable and has a configured MAC address, attempt wakeup
        if getattr(self, "wol_enabled", True):
            unreachable_nodes = [
                n
                for r, n in zip(server_reports, self.pool.nodes)
                if r.get("status") == "unreachable" and getattr(n, "mac", None)
            ]
            woken_nodes = []
            for n in unreachable_nodes:
                msg = f"Server '{n.label}' ({n.url}) unreachable. Sending Wake-on-LAN magic packet to {n.mac}..."
                if on_status_callback:
                    on_status_callback(msg)
                logger.info(msg)
                try:
                    if n.wake():
                        woken_nodes.append(n)
                except Exception as ex:
                    err_msg = f"Failed to send Wake-on-LAN magic packet to {n.label} ({n.mac}): {ex}"
                    logger.warning(err_msg)
                    if on_status_callback:
                        on_status_callback(err_msg)

            if woken_nodes:
                wait_seconds = getattr(self, "wol_wait_seconds", 120)
                probe_interval = getattr(self, "wol_probe_interval", 5.0)
                wait_msg = f"Wake-on-LAN sent to {len(woken_nodes)} server(s). Waiting up to {wait_seconds}s for boot..."
                if on_status_callback:
                    on_status_callback(wait_msg)
                logger.info(wait_msg)

                start_time = time.time()
                responsive_urls = set()
                while (time.time() - start_time) < wait_seconds:
                    time.sleep(probe_interval)
                    elapsed = int(time.time() - start_time)
                    for n in woken_nodes:
                        if n.url in responsive_urls:
                            continue
                        try:
                            _fetch_server_json(n.url, "/api/version")
                            responsive_urls.add(n.url)
                            rec_msg = f"Server '{n.label}' ({n.url}) is awake and responding ({elapsed}s)! Retrying warmup..."
                            if on_status_callback:
                                on_status_callback(rec_msg)
                            logger.info(rec_msg)
                        except Exception:
                            pass

                    if len(responsive_urls) == len(woken_nodes):
                        break

                if responsive_urls:
                    new_reports = []
                    for r, n in zip(server_reports, self.pool.nodes):
                        if n.url in responsive_urls:
                            new_reports.append(_inspect_node(n))
                        else:
                            new_reports.append(r)
                    server_reports = new_reports
                else:
                    timeout_msg = f"Woken server(s) did not respond within {wait_seconds}s. Continuing with available servers."
                    if on_status_callback:
                        on_status_callback(timeout_msg)
                    logger.warning(timeout_msg)

        primary_node = self.pool.primary_server
        primary_report = next((r for r in server_reports if r["url"] == primary_node.url), server_reports[0])

        return {
            "status": "ok",
            "model": self.model_name,
            "servers": server_reports,
            "primary": primary_report,
            "pool_status": self.pool.get_status(),
            "device": primary_report.get("device", "Unknown"),
            "is_gpu": primary_report.get("is_gpu", False),
            "size": primary_report.get("size", 0),
            "size_vram": primary_report.get("size_vram", 0),
            "vram_pct": primary_report.get("vram_pct", 0.0),
            "runner": primary_report.get("runner", "llamacpp"),
            "warning": primary_report.get("warning"),
        }
