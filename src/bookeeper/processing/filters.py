"""
Stateless Chunk Filtering Conveyor for Document Chunk Processing.

Provides an extensible, protocol-based conveyor pipeline with a 3-tier decision model:
  - SKIP: Drop chunk immediately; halts further conveyor processing.
  - RETAIN: Retain chunk immediately; halts further conveyor processing.
  - CONTINUE: Pass chunk to the next filter in the conveyor.

If the conveyor reaches the end with CONTINUE (or if the conveyor filter list is empty),
the chunk is retained (processed).
"""

from __future__ import annotations

import logging
import re
from abc import ABC, abstractmethod
from enum import Enum
from typing import Any, List, Optional, Protocol, Sequence, Tuple, runtime_checkable

logger = logging.getLogger(__name__)


class FilterDecision(str, Enum):
    """Three-tier decision returned by stateless chunk filters."""
    SKIP = "skip"        # Drop chunk immediately; stop further conveyor evaluation
    RETAIN = "retain"    # Keep chunk immediately; stop further conveyor evaluation
    CONTINUE = "continue"# Pass chunk to next filter in conveyor


# ==============================================================================
# 1. Regex Heuristics for Literature & Technical Filtering
# ==============================================================================

# LaTeX, Unicode Math, Operators & Equation Formulations
FORMULA_REGEX = re.compile(
    r"(?:"
    r"\\[a-zA-Z]+|"  # LaTeX commands: \frac, \sum, \alpha, \sim, \phi, etc.
    r"\$\$[\s\S]+?\$\$|" # Display math delimiters $$ ... $$
    r"\$[^$\n]{1,300}\$|" # Inline math delimiters $ ... $
    r"\\\[[\s\S]+?\\\]|" # Display math \[ ... \]
    r"\\\(.+?\\\)|\b(?:\\\(|\\\))|" # Inline math \( ... \)
    r"\b[a-zA-Z]_(?:\{[^{}]+\}|[a-zA-Z0-9]\b)|(?<=[\)\}\]])_(?:\{[^{}]+\}|[a-zA-Z0-9]\b)|" # Subscripts: x_{ij}, x_0, k_t, H_w
    r"\b[a-zA-Z]\^(?:\{[^{}]+\}|[a-zA-Z0-9\*\+⊤⊥\-]\b)|(?<=[\)\}\]])\^(?:\{[^{}]+\}|[a-zA-Z0-9\*\+⊤⊥\-]\b)|" # Superscripts: y^2, x^{n+1}, W^T, P^T
    r"[∑∫∏∂∇√αβγδεζηθικλμνξοπρστυφχψωΑΒΓΔΕΖΗΘΙΚΛΜΝΞΟΠΡΣΤΥΦΧΨΩ≤≥≠≈≡≢∈∉∀∃⊂⊃⊆⊇∪∩∼∝⊤⊥∠∧∨⊕⊗⊙⊘⊚⊢⊣⊨⊩⊪⋅⋆°′″‴…⋮⋯⋱≜≝≐≒≓≔≕]|" # Greek & Unicode math
    r"[ℝℤℕℚℂ𝔼ℙℒℋℱ𝒟]|" # Blackboard bold & math script
    r":=|≔|\bcoloneqq\b|≜|\btriangleq\b|" # Definition operators
    r"\b(?:tr|Cov|Var|argmax|argmin|arg\s*max|arg\s*min|diag|det|sup|inf|log|exp|softmax|sigmoid|ReLU|GeLU)\s*[\(\[_]|" # Math operators with args
    r"[A-Za-z0-9_,\(\)\{\}\[\]\^]+\s*(?:=|≈|~|:=|≔|≤|<=|≥|>=)\s*(?:E\s*[\[(]|\\sum|\\int|\\prod|tr\b|[0-9A-Za-z_\\\(])|" # Explicit equation / equivalence
    r"\b(?:Theorem|Lemma|Proposition|Corollary|Axiom)\s+\d+[:.]|" # Formal mathematical theorems/lemmas
    r"\b(?:mod|pmod)\s+[0-9a-zA-Z]+|\bcongruent\s+to\s+[0-9a-zA-Z]+|" # Modular congruence
    r"\b(?:quasiconvex|quasi-convex|Lipschitz|Hessian|eigenvalue|eigenvector|spectral\s+bound)\b" # Math / optimization concepts
    r")",
    re.IGNORECASE,
)

# Reference / Bibliography / Acknowledgments / Governance Section Titles
REFERENCE_SECTION_TITLE_REGEX = re.compile(
    r"(?:^|\b)(?:"
    r"references?|bibliography|works\s+cited|literature\s+cited|"
    r"further\s+reading|selected\s+references|bibliographic\s+notes|"
    r"acknowledg(?:e)?ments?|author\s+contributions?|competing\s+interests?|"
    r"funding|conflict\s+of\s+interests?|declarations?|data\s+availability|"
    r"code\s+availability|broader\s+impacts?|checklist|"
    r"reproducibility\s*statement|ethics\s*statement|ethical\s+considerations|"
    r"ai\s*use\s*statement|ai\s*disclosure|appendix\s*contents|table\s*of\s*contents"
    r")(?:\b|$)",
    re.IGNORECASE,
)

# Guard preserving active conceptual / methodological claims from aggressive boilerplate drops
PROSE_GUARD_REGEX = re.compile(
    r"\b(?:"
    r"we\s+(?:propose|introduce|present|show|demonstrate|evaluate|define|observe|find|use|train|develop|analyze|study|derive|formulate|ablate|investigate|compare|report|measure|test|conduct|construct|design)|"
    r"our\s+(?:method|approach|model|framework|experiments|results|analysis|algorithm|architecture|study|findings|benchmark)|"
    r"this\s+(?:section|chapter|paper|appendix|table|figure)\s+(?:presents|describes|shows|details|reports|evaluates|introduces|investigates|analyzes)|"
    r"in\s+this\s+(?:section|chapter|paper|work|study|appendix)|"
    r"theorem\s+\d|lemma\s+\d|proposition\s+\d|corollary\s+\d|definition\s+\d"
    r")\b",
    re.IGNORECASE,
)

# Governance, Ethics, AI Use, Reproducibility, Funding & Conference Checklists
GOVERNANCE_ETHICS_FUNDING_REGEX = re.compile(
    r"(?:"
    r"^(?:\d+(?:\.\d+)*\s+)?(?:acknowledg(?:e)?ments?|ethics\s*statement|ethical\s+considerations|reproducibility\s*statement|ai\s*use\s*statement|ai\s*disclosure|broader\s+impacts?|funding|competing\s+interests?)\b|"
    r"\backnowledge\s+support\s+from\b|\bsupported\s+in\s+part\s+by\s+grant\b|\bcontributed\s+equally\s+to\s+this\s+work\b|"
    r"\bno\s+additional\s+ethical\s+concerns\b|\bofficial\s+policies\s+or\s+endorsements\s+of\b|"
    r"\bneither\s+the\s+european\s+union\s+nor\b|\bvaluable\s+comments\s+on\s+an\s+earlier\s+draft\b|"
    r"\bAnswer:\s*\[(?:Yes|No|NA)\]\s*\n\s*Justification:\s*See\b"
    r")",
    re.IGNORECASE | re.DOTALL,
)

# Table of Contents & Appendix Directories (Leader Dots)
TOC_LEADER_DOTS_REGEX = re.compile(
    r"(?:(?:\.\s+){4,}\.|\.{5,}|(?:^|\n)\s*(?:APPENDIX\s*CONTENTS|TABLE\s*OF\s*CONTENTS)\b)",
    re.IGNORECASE,
)

CITATION_BRACKET_REGEX = re.compile(r"\[\d{1,3}\]")
CITATION_AUTHOR_YEAR_REGEX = re.compile(
    r"\b(?:[A-Z][a-zA-Z]+(?:\s+et\s+al\.?)?,?\s*(?:19|20)\d{2}[a-z]?)\b"
)
DOI_REGEX = re.compile(r"\b(?:doi(?::|\.org/)|10\.\d{4,9}/[-._;()/:A-Za-z0-9]+)\b", re.IGNORECASE)
ARXIV_ID_REGEX = re.compile(r"\barXiv:\d{4}\.\d{4,5}(?:v\d+)?\b", re.IGNORECASE)
URL_COUNT_REGEX = re.compile(r"https?://[^\s<>\"']+", re.IGNORECASE)
VENUE_REGEX = re.compile(
    r"(?:\b(?:"
    r"In\s*Proceedings\s+of|In\s*Proceedings|Proceedings\s+of|In\s*Findings\s+of|In\s*Advances\s+in|"
    r"Conference\s+on|Symposium\s+on|IEEE\s+Transactions|ACM\s+Transactions|"
    r"Transactions\s+on\s+Machine\s+Learning\s+Research|Journal\s+of\s+[A-Z][a-z]+|"
    r"Association\s+for\s+Computational\s+Linguistics|arXiv\s+preprint|"
    r"NeurIPS|ICML|ICLR|CVPR|ECCV|ICCV|ACL|EMNLP|NAACL|KDD|AAAI|SIGIR|IJCAI|PMLR|JMLR|"
    r"MIT\s+Press|Oxford\s+University\s+Press|Curran\s+Associates|OpenReview\.net|"
    r"ISSN\s+\d{4}-\d{4}|ISBN\s+[\d\-]+|"
    r"vol\.\s*\d+|volume\s+\d+|no\.\s*\d+|pp\.\s*\d+[–\-—]+\d+|pages\s+\d+[–\-—]+\d+"
    r")\b|"
    r"https?://(?:arxiv\.org|openreview\.net|aclanthology\.org|doi\.org|proceedings\.mlr\.press|proceedings\.iclr\.cc)|\b\d+\(\d+\):\d+[–\-—]+\d+)",
    re.IGNORECASE,
)
SHORT_REF_KEYWORD_REGEX = re.compile(
    r"\b(?:In\s*Proceedings|arXiv(?:\s+preprint)?|doi(?::|\.org/)|eds?\.|vol\.\s*\d+|pp\.\s*\d+[–\-—]+\d+|pages\s+\d+[–\-—]+\d+|URL\s*https?://)\b",
    re.IGNORECASE,
)
BIB_ENTRY_LINE_REGEX = re.compile(
    r"^[ \t]*(?:\[\d{1,3}\]|\d{1,3}\.\s+[A-Z]|[A-Z][a-zA-Z´'\-]+,\s+[A-Z]\.|\b(?:et\s+al\.|ed\.|eds\.)\b)",
    re.MULTILINE,
)

# Technical Boilerplate, Title/Affiliation Blocks & Licenses
CODE_HEADER_BOILERPLATE_REGEX = re.compile(
    r"(?:#include\s+<[a-z0-9_./]+>|#define\s+[A-Z0-9_]+|#ifndef\s+[A-Z0-9_]+|#endif)",
    re.IGNORECASE,
)
LICENSE_BOILERPLATE_REGEX = re.compile(
    r"(?:All\s+rights\s+reserved|GNU\s+General\s+Public\s+License|Apache\s+License,\s+Version|Permission\s+is\s+hereby\s+granted,\s+free\s+of\s+charge|MIT\s+License|Creative\s+Commons\s+Attribution)",
    re.IGNORECASE,
)
WEB_SCRAPE_NOISE_REGEX = re.compile(
    r"(?:Posted\s+by\s*[:\w]+|Reply\s+with\s+quote|Join\s+Date\s*:\s*\w+|Posts\s*:\s*\d+|Location\s*:\s*\w+)",
    re.IGNORECASE,
)
TITLE_AFFILIATION_HEADER_REGEX = re.compile(
    r"(?:\bCorrespondence\s+to:|\b1\s*[A-Z][a-zA-Z\s,]+(?:University|Institute|Center|College|Laboratory)\b)",
    re.IGNORECASE,
)

# Dataset Stimulus, Verbatim Prompt Templates, Chat Wrappers & JSON Schemas
SURVEY_QUESTION_REGEX = re.compile(
    r"(?:(?:Question|Item)\s+\d+[:.]|\bOption\s+[A-D][:.]|\b[A-D]\)\s+[A-Z]|\bLikert\s+scale\b)",
    re.IGNORECASE,
)
TOKENIZER_NOISE_REGEX = re.compile(r"@-@|@\.@|<unk>", re.IGNORECASE)
PROMPT_TEMPLATE_SCHEMA_REGEX = re.compile(
    r"(?:^<\|im_start\|>|<\|im_start\|>[\s\S]*?<\|im_end\|>|<curr_prompt>|\"zero_overlap_verified\":\s*true|"
    r"\binstruction\s*:\s*\.\.\.\.?,?\s*output\s*:\s*\.\.\.)",
    re.IGNORECASE,
)

# Standalone Captions & Short Cross-Reference Signposts
STANDALONE_CAPTION_REGEX = re.compile(
    r"^\([a-d]\)\s+[A-Z].*?\([b-e]\)\s+[A-Z]",
    re.IGNORECASE | re.DOTALL,
)
CROSS_REF_SIGNPOST_REGEX = re.compile(
    r"(?:\bAll\s+proofs\s+are\s+provided\s+in\s+Appendix\b|"
    r"\bPlot\s+conventions\s+appear\s+in\s+Appendix\b|"
    r"\bTables?\s+\d+[–\-—]?\d*\s+list\s+the\b.*?\bbehind\s+Figures?\b|"
    r"\b(?:are\s+provided\s+in|appear\s+in|can\s+be\s+found\s+in)\s+Appendix\s+[A-Z0-9.\-]+)",
    re.IGNORECASE | re.DOTALL,
)


# ==============================================================================
# 2. Chunk Attribute Access & Mutation Helpers
# ==============================================================================

def get_chunk_attr(chunk: Any, attr: str, default: Any = None) -> Any:
    """Safely get an attribute from a chunk whether it is an object or dictionary."""
    if isinstance(chunk, dict):
        return chunk.get(attr, default)
    return getattr(chunk, attr, default)


def set_chunk_attr(chunk: Any, attr: str, value: Any) -> None:
    """Safely set an attribute on a chunk whether it is an object or dictionary."""
    if isinstance(chunk, dict):
        chunk[attr] = value
    else:
        try:
            setattr(chunk, attr, value)
        except Exception:
            pass


def extract_chunk_text_and_titles(chunk: Any) -> Tuple[str, str, str, str]:
    """Helper to safely extract text, section_title, chapter_title, and breadcrumb."""
    if isinstance(chunk, dict):
        text = str(chunk.get("text", "") or "").strip()
        sec = str(chunk.get("section_title", "") or "").strip()
        ch = str(chunk.get("chapter_title", "") or "").strip()
        crumb = str(chunk.get("breadcrumb", "") or "").strip()
    else:
        text = str(getattr(chunk, "text", "") or "").strip()
        sec = str(getattr(chunk, "section_title", "") or "").strip()
        ch = str(getattr(chunk, "chapter_title", "") or "").strip()
        crumb = str(getattr(chunk, "breadcrumb", "") or "").strip()
    return text, sec, ch, crumb


# Pre-screening patterns for URL query strings and currency amounts before formula detection
_URL_STRIP_REGEX = re.compile(r"https?://[^\s<>\"']+", re.IGNORECASE)
_CURRENCY_STRIP_REGEX = re.compile(
    r"(?<!\\)\$\d+(?:,\d{3})*(?:\.\d+)?(?:\s*(?:million|billion|trillion|k|M|B)\b)?",
    re.IGNORECASE,
)


def has_mathematical_formula(text: str) -> bool:
    """
    Detect whether text contains mathematical equations or derivations.
    Pre-strips URLs (e.g. openreview.net/forum?id=...) and currency amounts ($0.10, $1)
    to prevent URL query parameters or dollar prices from triggering false formula matches.
    """
    if not text:
        return False
    cleaned = _URL_STRIP_REGEX.sub(" ", text)
    cleaned = _CURRENCY_STRIP_REGEX.sub(" ", cleaned)
    return bool(FORMULA_REGEX.search(cleaned))


# ==============================================================================
# 3. Stateless Chunk Filter Interface & Implementations
# ==============================================================================

@runtime_checkable
class ChunkFilter(Protocol):
    """
    Stateless filter interface for chunk evaluation conveyor.
    Evaluates a chunk, optionally mutates chunk attributes (e.g. has_formula),
    and returns a FilterDecision (SKIP, RETAIN, or CONTINUE).
    """

    def verify(self, chunk: Any) -> FilterDecision:
        ...


class BaseChunkFilter(ABC):
    """Abstract base class for stateless chunk filters."""
    name: str = "filter"

    @abstractmethod
    def verify(self, chunk: Any) -> FilterDecision:
        """Evaluate a chunk and return SKIP, RETAIN, or CONTINUE."""
        pass


class FormulaDetectionFilter(BaseChunkFilter):
    """
    Detects mathematical formulas in chunks and updates chunk.has_formula.
    Returns RETAIN if chunk contains a mathematical formula and retain_on_formula=True,
    stopping further filtering and retaining the chunk immediately.
    Otherwise returns CONTINUE so subsequent filters know chunk.has_formula is True.
    """
    name: str = "formulas"

    def __init__(self, retain_on_formula: bool = True):
        self.retain_on_formula = retain_on_formula

    def verify(self, chunk: Any) -> FilterDecision:
        text, _, _, _ = extract_chunk_text_and_titles(chunk)
        existing = get_chunk_attr(chunk, "has_formula", None)
        if existing is None:
            is_formula = has_mathematical_formula(text)
            set_chunk_attr(chunk, "has_formula", is_formula)
        elif not existing:
            is_formula = has_mathematical_formula(text)
            if is_formula:
                set_chunk_attr(chunk, "has_formula", True)
        else:
            # Re-verify if existing was set from raw URL/currency false positive
            is_formula = has_mathematical_formula(text) if text else True
            set_chunk_attr(chunk, "has_formula", is_formula)

        if is_formula and text:
            # Guard: if the only formula match is a weak symbol (e.g. prime ′, ∼, ×) inside a pure bibliography slice,
            # do not shield it from BibliographicFilter
            cleaned = _CURRENCY_STRIP_REGEX.sub(" ", _URL_STRIP_REGEX.sub(" ", text))
            matches = [m.strip() for m in FORMULA_REGEX.findall(cleaned)]
            if matches and all(m in ("′", "″", "∼", "×", "⋅", "°", "…") for m in matches):
                if BibliographicFilter().verify(chunk) == FilterDecision.SKIP:
                    is_formula = False
                    set_chunk_attr(chunk, "has_formula", False)

        if is_formula and self.retain_on_formula:
            return FilterDecision.RETAIN
        return FilterDecision.CONTINUE


class EmptyChunkFilter(BaseChunkFilter):
    """Rejects chunks that are empty or pure whitespace."""
    name: str = "empty"

    def verify(self, chunk: Any) -> FilterDecision:
        text, _, _, _ = extract_chunk_text_and_titles(chunk)
        if not text or not text.strip():
            return FilterDecision.SKIP
        return FilterDecision.CONTINUE


class ReferenceSectionFilter(BaseChunkFilter):
    """
    Rejects chunks originating from reference, bibliography, acknowledgment,
    governance (AI disclosure, reproducibility, ethics, funding, checklists),
    or table-of-contents directory sections.
    """
    name: str = "ref_sections"

    def verify(self, chunk: Any) -> FilterDecision:
        text, sec_title, ch_title, breadcrumb = extract_chunk_text_and_titles(chunk)
        for cand in (sec_title, ch_title, breadcrumb):
            if cand and REFERENCE_SECTION_TITLE_REGEX.search(cand):
                return FilterDecision.SKIP

        lines = [l.strip() for l in text.split("\n") if l.strip()]
        first_line = lines[0].lower() if lines else ""
        if REFERENCE_SECTION_TITLE_REGEX.match(first_line):
            return FilterDecision.SKIP

        # Table of Contents & Appendix Directories with leader dots (. . . . 24)
        if TOC_LEADER_DOTS_REGEX.search(text) and len(text) < 900:
            return FilterDecision.SKIP

        # Governance, AI Use, Reproducibility, Ethics, Funding & Conference Checklists (strictly short blocks)
        if GOVERNANCE_ETHICS_FUNDING_REGEX.search(text.strip()) and len(text) < 320:
            if not PROSE_GUARD_REGEX.search(text):
                return FilterDecision.SKIP

        return FilterDecision.CONTINUE


class BibliographicFilter(BaseChunkFilter):
    """Rejects citation dumps and bibliography blocks while preserving technical prose, algorithms, and preprints."""
    name: str = "biblio"

    def verify(self, chunk: Any) -> FilterDecision:
        text, _, _, _ = extract_chunk_text_and_titles(chunk)
        if not text:
            return FilterDecision.SKIP

        t_len = len(text)
        lines = [l.strip() for l in text.split("\n") if l.strip()]
        bracket_cites = len(CITATION_BRACKET_REGEX.findall(text))
        dois = len(DOI_REGEX.findall(text))
        arxiv = len(ARXIV_ID_REGEX.findall(text))
        bib_l = len(BIB_ENTRY_LINE_REGEX.findall(text))
        venues = len(VENUE_REGEX.findall(text))

        # 1. High density of bracket citations ([1], [2], [3]...)
        if bracket_cites >= 5 and len(lines) <= bracket_cites + 3:
            return FilterDecision.SKIP
        if bracket_cites >= 8:
            return FilterDecision.SKIP

        # 2. Bibliography entries listing papers with venues, DOIs, or arXiv IDs
        if (dois >= 1 or arxiv >= 1 or venues >= 1) and bib_l >= 2 and len(lines) <= (bib_l + 3):
            return FilterDecision.SKIP

        # 3. Isolated single arXiv citation slice listing author names
        if bool(re.search(r"arXiv:\d{4}\.\d{4,5}.*?[A-Z][a-z]+", text, re.S)) and len(lines) <= 5:
            return FilterDecision.SKIP

        # 4. Isolated proceedings citation slice
        if len(lines) <= 4 and bool(
            re.search(r"^(?:In\s*Proceedings|In\s*Findings|In\s*Advances|Proceedings\s+of)\b.*?(?:pages|pp\.)\s*\d+[–\-—]+\d+", text.strip(), re.I | re.S)
        ):
            return FilterDecision.SKIP

        # 5. Pure author list fragment (comma-separated list of 4+ names in <= 3 lines)
        if len(lines) <= 3:
            author_tokens = re.findall(r"\b[A-Z][a-zA-Z´'\-]+(?:\s+[A-Z]\.|\s+[A-Z][a-zA-Z´'\-]+)?,", text)
            if len(author_tokens) >= 4 and t_len < 350:
                return FilterDecision.SKIP

        # 6. Inline References / Bibliography transition header followed by citations
        if bool(re.search(r"(?:^|\n|\.\s+)(?:References|REFERENCES|Bibliography|BIBLIOGRAPHY)\s*\n", text)) and t_len < 650:
            return FilterDecision.SKIP

        # 7. High-precision pure author lists & venue/DOI citation tails without 'Concept: Subtitle' definitions
        if not PROSE_GUARD_REGEX.search(text):
            author_tokens = re.findall(r"\b[A-Z][a-zA-Z´'\-]+(?:\s+[A-Z]\.|\s+[A-Z][a-zA-Z´'\-]+)?,", text)
            text_no_urls = re.sub(r"https?://\S+|doi:\s*\S+|arXiv:\s*\S+|\b\d+\(\d+\):\d+", "", text, flags=re.I)
            if ":" not in text_no_urls and "?" not in text_no_urls and t_len < 280:
                if len(lines) <= 4 and len(author_tokens) >= 5:
                    return FilterDecision.SKIP
                if t_len < 220 and bool(
                    re.match(
                        r"^(?:In\s*Proceedings|In\s*Findings|In\s*Advances|Proceedings\s+of|Association\s+for|Asso-\s*ciation\s+for|volume\s+\d+|vol\.\s*\d+|pages\s+\d+|pp\.\s*\d+|\d+\s*[–\-—]\s*\d+|doi:|URL\s*https?://)",
                        text.strip(),
                        re.I,
                    )
                ):
                    return FilterDecision.SKIP
                if (
                    t_len < 250
                    and (dois >= 1 or arxiv >= 1 or venues >= 1)
                    and bool(re.search(r"\b(?:19|20)\d{2}[a-z]?\b", text))
                    and (
                        bool(re.search(r"(?:^\s*\[\d{1,3}\]|(?:^|\n)\s*\d{1,3}\.\s+[A-Z][a-zA-Z\-]+,\s*[A-Z]\.)", text))
                        or len(author_tokens) >= 2
                    )
                ):
                    return FilterDecision.SKIP

        return FilterDecision.CONTINUE


class TechnicalBoilerplateFilter(BaseChunkFilter):
    """Rejects pure C/C++ compilation headers, standalone license files, author affiliation blocks, and running page headers."""
    name: str = "tech_boiler"

    def verify(self, chunk: Any) -> FilterDecision:
        text, _, _, _ = extract_chunk_text_and_titles(chunk)
        if not text:
            return FilterDecision.SKIP

        t_len = len(text)
        lines = [l.strip() for l in text.split("\n") if l.strip()]
        code_headers = len(CODE_HEADER_BOILERPLATE_REGEX.findall(text))
        if code_headers >= 3 and len(lines) <= code_headers + 3:
            return FilterDecision.SKIP

        if bool(LICENSE_BOILERPLATE_REGEX.search(text)) and t_len < 400:
            return FilterDecision.SKIP

        if bool(WEB_SCRAPE_NOISE_REGEX.search(text)) and len(lines) <= 5:
            return FilterDecision.SKIP

        # Title / Author Affiliation Blocks (strictly short blocks without Abstract or conceptual claims)
        if t_len < 300 and not re.search(r"\bAbstract\b", text, re.I) and not PROSE_GUARD_REGEX.search(text):
            if bool(TITLE_AFFILIATION_HEADER_REGEX.search(text)):
                return FilterDecision.SKIP

        return FilterDecision.CONTINUE


class DatasetStimulusFilter(BaseChunkFilter):
    """
    Rejects survey questionnaires, benchmark tokenizer noise, verbatim LLM prompt wrappers,
    JSON agent config schemas, and dense VQA question dumps.
    """
    name: str = "dataset_stim"

    def verify(self, chunk: Any) -> FilterDecision:
        text, sec_title, _, _ = extract_chunk_text_and_titles(chunk)
        if not text:
            return FilterDecision.SKIP

        t_len = len(text)
        if bool(TOKENIZER_NOISE_REGEX.search(text)):
            return FilterDecision.SKIP

        if sec_title and re.search(r"^(?:appendix:?\s*)?(?:survey\s+questionnaire|raw\s+questionnaire|survey\s+items)$", sec_title.strip(), re.I):
            return FilterDecision.SKIP

        lines = [l.strip() for l in text.split("\n") if l.strip()]
        q_matches = len(re.findall(r"(?:Question|Item)\s+\d+[:.]", text, re.I))
        opt_matches = len(re.findall(r"\bOption\s+[A-D][:.]", text, re.I))
        if q_matches >= 1 and opt_matches >= 3 and len(lines) <= q_matches + opt_matches + 2:
            return FilterDecision.SKIP

        # Verbatim LLM chat wrappers (<|im_start|>...<|im_end|>) and short JSON agent schemas
        if t_len < 400 and not PROSE_GUARD_REGEX.search(text) and bool(PROMPT_TEMPLATE_SCHEMA_REGEX.search(text.strip())):
            return FilterDecision.SKIP

        # Dense VQA benchmark evaluation question lists (4+ question marks in a very short < 150 char fragment)
        if text.count("?") >= 4 and t_len < 150 and not PROSE_GUARD_REGEX.search(text):
            return FilterDecision.SKIP

        return FilterDecision.CONTINUE


class DiagramOCRFilter(BaseChunkFilter):
    """Rejects genuine OCR layout garbage, checkmark benchmark matrices, and diagram step callout lists."""
    name: str = "diag_ocr"

    def verify(self, chunk: Any) -> FilterDecision:
        text, _, _, _ = extract_chunk_text_and_titles(chunk)
        if not text:
            return FilterDecision.SKIP

        # If chunk is verified to have mathematical formulas, do not drop
        if get_chunk_attr(chunk, "has_formula", False):
            return FilterDecision.CONTINUE

        # Raw checkmark / cross benchmark comparison tables (e.g. × × × × ✓ ✓)
        if bool(re.search(r"(?:[×✓✗✔✘]\s*){4,}", text)) and len(text) < 600:
            return FilterDecision.SKIP

        # Numbered diagram / exhibit step callout lists (01 Label ... 02 Label ... 03 Label ... 04 Label)
        if bool(re.search(r"(?:^|\n)\s*01\s+[A-Z][a-z]+.*?\s+02\s+[A-Z][a-z]+.*?\s+03\s+[A-Z][a-z]+.*?\s+04\s+[A-Z][a-z]+", text, re.S)) and len(text) < 450:
            return FilterDecision.SKIP

        lines = [l.strip() for l in text.split("\n") if l.strip()]
        if len(lines) >= 8:
            avg_len = sum(len(l) for l in lines) / len(lines)
            # Check for markdown table structure
            has_table_sep = any(re.match(r"^\|?[\s:-]+\|[\s:-|]+$", l) for l in lines)
            # Check for code/algorithm constructs
            has_code_keywords = any(re.search(r"\b(def|class|return|for\s+.*\s+in|while|if\s+.*:)\b", l) for l in lines)
            # Count sentence terminators
            sentence_ends = sum(1 for l in lines if l.endswith(".") or l.endswith(";"))

            if not has_table_sep and not has_code_keywords and sentence_ends <= 1:
                short_line_ratio = sum(1 for l in lines if len(l) <= 45) / len(lines)
                if avg_len <= 22 and short_line_ratio >= 0.85:
                    return FilterDecision.SKIP

        return FilterDecision.CONTINUE


class CaptionAndSignpostFilter(BaseChunkFilter):
    """
    Rejects short standalone Figure/Table/Listing captions (< 140 chars without explanatory verbs),
    subfigure labels ((a) ... (b) ... < 200 chars), and short cross-reference signpost chunks (< 160 chars)
    that merely point to Appendices, Figures, or Tables without conceptual claims.
    Executed before IdeaSimilarityFilter and ChunkSimilarityFilter.
    """
    name: str = "captions_signposts"

    def verify(self, chunk: Any) -> FilterDecision:
        text, _, _, _ = extract_chunk_text_and_titles(chunk)
        if not text:
            return FilterDecision.SKIP

        if get_chunk_attr(chunk, "has_formula", False):
            return FilterDecision.CONTINUE

        t_len = len(text)

        if t_len < 200 and not PROSE_GUARD_REGEX.search(text):
            # 1. Subfigure label block ((a) ... (b) ...)
            if bool(STANDALONE_CAPTION_REGEX.search(text.strip())):
                return FilterDecision.SKIP

            # 2. Short standalone Figure/Table/Listing caption without explanatory prose verbs
            if t_len < 140 and bool(re.match(r"^(?:Figure|Fig\.|Table|Tab\.|Listing)\s*\d+[:.]", text.strip(), re.I)):
                if not bool(re.search(r"\b(?:shows?|demonstrates?|illustrates?|compares?|presents?|reports?|reveals?|indicates?|outperforms?|achieves?|improves?)\b", text, re.I)):
                    return FilterDecision.SKIP

            # 3. Short Cross-Reference Pointers & Section Signposts
            if t_len < 160 and bool(CROSS_REF_SIGNPOST_REGEX.search(text)):
                return FilterDecision.SKIP

        return FilterDecision.CONTINUE


class ChunkSimilarityFilter(BaseChunkFilter):
    """
    Drops repetitive / redundant narrative chunks within a section based on cosine similarity
    against recently retained section chunks (Hierarchical Semantic Cover).
    """
    name: str = "chunk_sim"

    def __init__(
        self,
        similarity_threshold: float = 0.92,
        sliding_window_size: int = 3,
        min_section_chunks: int = 2,
    ):
        self.similarity_threshold = float(similarity_threshold)
        self.sliding_window_size = int(sliding_window_size)
        self.min_section_chunks = int(min_section_chunks)
        self._section_windows: dict[Any, list[Any]] = {}
        self._section_chunk_counts: dict[Any, int] = {}

    def reset(self) -> None:
        """Reset internal section window tracking."""
        self._section_windows.clear()
        self._section_chunk_counts.clear()

    def verify(self, chunk: Any) -> FilterDecision:
        emb = getattr(chunk, "embedding", None)
        if emb is None or len(emb) <= 1:
            return FilterDecision.CONTINUE

        from pathlib import Path
        import numpy as np

        arr = np.asarray(emb, dtype=np.float32)
        if arr.ndim != 1:
            arr = arr.flatten()
        if arr.shape[0] != 768 and arr.shape[0] == 1024 and Path("output/jepa_768_weights.npz").is_file():
            try:
                from bookeeper.evaluation.jepa_encoder import LightweightJEPAEncoder
                if not hasattr(self, "_jepa_proj"):
                    self._jepa_proj = LightweightJEPAEncoder.load("output/jepa_768_weights.npz")
                arr = self._jepa_proj.project(arr)
            except Exception:
                pass

        norm = float(np.linalg.norm(arr))
        if norm < 1e-12:
            return FilterDecision.CONTINUE
        v_unit = arr / norm

        sec_key = (
            getattr(chunk, "chapter_idx", 0),
            str(getattr(chunk, "section_title", "") or "").strip().lower(),
        )

        cnt = self._section_chunk_counts.get(sec_key, 0)
        self._section_chunk_counts[sec_key] = cnt + 1

        window = self._section_windows.setdefault(sec_key, [])
        if not window:
            # First non-reference chunk of the section is always the section anchor
            window.append(v_unit)
            return FilterDecision.CONTINUE

        # Compare against recently retained chunks in this section (within sliding window)
        compat_window = [w for w in window[-self.sliding_window_size:] if w.shape == v_unit.shape]
        if not compat_window:
            max_sim = -1.0
        else:
            W = np.stack(compat_window, axis=0)
            sims = W @ v_unit
            max_sim = float(np.max(sims))

        # Formula safeguard: Math formulas have absolute highest priority - NEVER drop as redundant!
        text = str(getattr(chunk, "text", "") or "")
        if get_chunk_attr(chunk, "has_formula", False) or has_mathematical_formula(text):
            set_chunk_attr(chunk, "has_formula", True)
            window.append(v_unit)
            set_chunk_attr(chunk, "similarity_to_cover", max_sim)
            return FilterDecision.CONTINUE

        # Length Guard: A short snippet/header (<300 chars) cannot cover a substantive chunk (>=500 chars)
        text_len = len(text)
        if max_sim >= self.similarity_threshold and text_len < 500:
            set_chunk_attr(chunk, "similarity_to_cover", max_sim)
            return FilterDecision.SKIP

        # Chunk introduces a distinct thought or subtopic shift
        window.append(v_unit)
        set_chunk_attr(chunk, "similarity_to_cover", max_sim)
        return FilterDecision.CONTINUE


class IdeaSimilarityFilter(BaseChunkFilter):
    """
    Evaluates chunks against an Idea Embedding Index:
      - similarity >= high_confidence_threshold: directly assigned (RETAIN, skips LLM).
      - unassigned_threshold <= similarity < high_confidence_threshold: limbo (SKIP, unless recovered by size).
      - similarity < unassigned_threshold: brand new idea (CONTINUE, passes to LLM).
    """
    name: str = "limbo"

    def __init__(
        self,
        idea_index: Optional[Any] = None,
        unassigned_threshold: float = 0.72,
        high_confidence_threshold: float = 0.87,
        skip_limbo_chunks: bool = True,
        max_batch_chunk_size: Optional[int] = 1024,
    ):
        self.idea_index = idea_index
        self.unassigned_threshold = float(unassigned_threshold)
        self.high_confidence_threshold = float(high_confidence_threshold)
        self.skip_limbo_chunks = bool(skip_limbo_chunks)
        self.max_batch_chunk_size = int(max_batch_chunk_size) if max_batch_chunk_size is not None else None

    def verify(self, chunk: Any) -> FilterDecision:
        if self.idea_index is None:
            return FilterDecision.CONTINUE

        emb = getattr(chunk, "embedding", None)
        if emb is None or len(emb) <= 1:
            return FilterDecision.CONTINUE

        # Query idea index for max similarity
        if hasattr(self.idea_index, "find_nearest"):
            best_idea, max_sim, best_summary = self.idea_index.find_nearest(emb)
        elif hasattr(self.idea_index, "find_top_k"):
            top = self.idea_index.find_top_k(emb, k=1)
            if top:
                best_idea, max_sim, best_summary = top[0][0], top[0][1], ""
            else:
                best_idea, max_sim, best_summary = None, 0.0, ""
        else:
            return FilterDecision.CONTINUE

        if best_idea is None:
            return FilterDecision.CONTINUE

        set_chunk_attr(chunk, "matched_idea", best_idea)
        set_chunk_attr(chunk, "matched_similarity", max_sim)
        set_chunk_attr(chunk, "matched_idea_summary", best_summary)

        # 1. High confidence direct match -> RETAIN (skip LLM extraction)
        if max_sim >= self.high_confidence_threshold:
            set_chunk_attr(chunk, "is_high_confidence", True)
            return FilterDecision.RETAIN

        # 2. Limbo chunk
        if self.skip_limbo_chunks and (max_sim >= self.unassigned_threshold):
            # Formula safeguard: Math formulas have absolute highest priority - NEVER skip as limbo!
            text_str = str(getattr(chunk, "text", "") or "")
            if get_chunk_attr(chunk, "has_formula", False) or has_mathematical_formula(text_str):
                set_chunk_attr(chunk, "has_formula", True)
                set_chunk_attr(chunk, "recovered_from_limbo", True)
                return FilterDecision.CONTINUE

            text_len = len(text_str)
            # If chunk is large enough (>= max_batch_chunk_size), recover it from limbo
            if self.max_batch_chunk_size is not None and text_len >= self.max_batch_chunk_size:
                set_chunk_attr(chunk, "recovered_from_limbo", True)
                return FilterDecision.CONTINUE
            # Otherwise drop/skip limbo chunk
            set_chunk_attr(chunk, "skipped_limbo", True)
            return FilterDecision.SKIP

        # 3. Unassigned / brand new idea -> CONTINUE to LLM
        return FilterDecision.CONTINUE


# ==============================================================================
# 4. Conveyor Pipeline Runner & Factory
# ==============================================================================

class ChunkFilterConveyor:
    """
    Ordered conveyor of stateless ChunkFilter instances.

    Execution semantics:
    1. Sequentially passes each chunk to filter.verify(chunk).
    2. If a filter returns SKIP -> immediately stops and returns SKIP (chunk dropped).
    3. If a filter returns RETAIN -> immediately stops and returns RETAIN (chunk kept).
    4. If a filter returns CONTINUE -> proceeds to next filter in conveyor.
    5. If all filters return CONTINUE (or conveyor filter list is empty) -> returns RETAIN (chunk kept).
    """

    def __init__(self, filters: Optional[Sequence[ChunkFilter]] = None):
        self.filters: List[ChunkFilter] = list(filters) if filters is not None else []

    def verify(self, chunk: Any) -> FilterDecision:
        """Run chunk through conveyor and return final FilterDecision."""
        for flt in self.filters:
            decision = flt.verify(chunk)
            if decision in (FilterDecision.SKIP, FilterDecision.RETAIN):
                return decision
        return FilterDecision.RETAIN

    def verify_with_category(self, chunk: Any) -> Tuple[FilterDecision, str]:
        """
        Run chunk through conveyor and return (decision, category).
        - If chunk contains formulas and retain_on_formula=True: (FilterDecision.RETAIN, "formulas")
        - If a filter drops the chunk: (FilterDecision.SKIP, filter.name)
        - If passed all filters: (FilterDecision.RETAIN, "passed")
        """
        for flt in self.filters:
            decision = flt.verify(chunk)
            if decision == FilterDecision.RETAIN:
                return (FilterDecision.RETAIN, getattr(flt, "name", "formulas"))
            elif decision == FilterDecision.SKIP:
                return (FilterDecision.SKIP, getattr(flt, "name", "filter"))
        return (FilterDecision.RETAIN, "passed")

    def should_retain(self, chunk: Any) -> bool:
        """Convenience method: True if chunk is retained, False if skipped."""
        return self.verify(chunk) != FilterDecision.SKIP

    def should_skip(self, chunk: Any) -> bool:
        """Convenience method: True if chunk is skipped, False if retained."""
        return self.verify(chunk) == FilterDecision.SKIP

    def filter_chunks(self, chunks: Sequence[Any]) -> List[Any]:
        """Filters a sequence of chunks, returning only retained chunks."""
        return [c for c in chunks if self.should_retain(c)]


def format_filter_stats(stats: dict[str, int]) -> str:
    """Format filter statistics dictionary into a clean, concise summary string."""
    order = [
        "formulas",
        "biblio",
        "diag_ocr",
        "captions_signposts",
        "ref_sections",
        "tech_boiler",
        "dataset_stim",
        "empty",
        "limbo",
        "chunk_sim",
    ]
    parts: list[str] = []
    for k in order:
        v = stats.get(k, 0)
        if v > 0:
            parts.append(f"{k}: {v}")
    for k, v in sorted(stats.items()):
        if k not in order and v > 0:
            parts.append(f"{k}: {v}")
    return ", ".join(parts) if parts else "none"


def format_short_drop_stats(stats: dict[str, int]) -> str:
    """Format short dropped categories for progress display."""
    order = [
        "biblio",
        "diag_ocr",
        "captions_signposts",
        "ref_sections",
        "tech_boiler",
        "dataset_stim",
        "empty",
        "limbo",
        "chunk_sim",
    ]
    short_names = {
        "biblio": "biblio",
        "diag_ocr": "diag_ocr",
        "captions_signposts": "captions",
        "ref_sections": "ref",
        "tech_boiler": "tech",
        "dataset_stim": "dataset",
        "empty": "empty",
        "limbo": "limbo",
        "chunk_sim": "sim",
    }
    parts: list[str] = []
    for k in order:
        v = stats.get(k, 0)
        if v > 0:
            label = short_names.get(k, k)
            parts.append(f"{label}: {v}")
    return ", ".join(parts)


def build_chunk_filter_conveyor(
    enable_formula_detection: bool = True,
    retain_on_formula: bool = True,
    skip_reference_sections: bool = True,
    skip_bibliographic: bool = True,
    skip_technical_boilerplate: bool = True,
    skip_dataset_stimulus: bool = True,
    skip_diagram_ocr: bool = True,
    skip_captions_and_signposts: bool = True,
    skip_empty: bool = True,
    # Chunk similarity filter (hierarchical cover drop)
    enable_chunk_similarity: bool = False,
    chunk_similarity_threshold: float = 0.92,
    chunk_window_size: int = 3,
    chunk_min_section_chunks: int = 2,
    # Idea similarity filter (semantic routing & limbo)
    enable_idea_similarity: bool = False,
    idea_index: Optional[Any] = None,
    unassigned_threshold: float = 0.72,
    high_confidence_threshold: float = 0.87,
    skip_limbo_chunks: bool = True,
    max_batch_chunk_size: Optional[int] = 1024,
) -> ChunkFilterConveyor:
    """
    Builds a ChunkFilterConveyor with filters enabled based on switcher flags.
    If a switcher is False, that filter is omitted from the conveyor.
    If all switchers are False, conveyor is empty and 100% of chunks are processed.
    """
    filters: List[ChunkFilter] = []

    # 1. Formula detection runs first to annotate chunk.has_formula for downstream filters
    if enable_formula_detection:
        filters.append(FormulaDetectionFilter(retain_on_formula=retain_on_formula))

    # 2. Basic empty guard
    if skip_empty:
        filters.append(EmptyChunkFilter())

    # 3. High-level section references, governance, and TOC directories
    if skip_reference_sections:
        filters.append(ReferenceSectionFilter())

    # 4. Bibliographic and citation slices
    if skip_bibliographic:
        filters.append(BibliographicFilter())

    # 5. Technical code headers, software licenses, and affiliation headers
    if skip_technical_boilerplate:
        filters.append(TechnicalBoilerplateFilter())

    # 6. Survey questions, dataset tokenizer noise, and prompt wrappers
    if skip_dataset_stimulus:
        filters.append(DatasetStimulusFilter())

    # 7. Diagram OCR layout dumps (checks chunk.has_formula before dropping)
    if skip_diagram_ocr:
        filters.append(DiagramOCRFilter())

    # 8. Standalone figure/table captions and short cross-reference signposts (before similarity filters)
    if skip_captions_and_signposts:
        filters.append(CaptionAndSignpostFilter())

    # 9. Idea similarity filter (Middle Priority: Semantic routing against graph ideas & limbo recovery)
    if enable_idea_similarity:
        filters.append(
            IdeaSimilarityFilter(
                idea_index=idea_index,
                unassigned_threshold=unassigned_threshold,
                high_confidence_threshold=high_confidence_threshold,
                skip_limbo_chunks=skip_limbo_chunks,
                max_batch_chunk_size=max_batch_chunk_size,
            )
        )

    # 10. Chunk-to-chunk semantic redundancy filter (Least Priority: Hierarchical Cover intra-section pruning)
    if enable_chunk_similarity:
        filters.append(
            ChunkSimilarityFilter(
                similarity_threshold=chunk_similarity_threshold,
                sliding_window_size=chunk_window_size,
                min_section_chunks=chunk_min_section_chunks,
            )
        )

    return ChunkFilterConveyor(filters)


_DEFAULT_CONVEYOR: Optional[ChunkFilterConveyor] = None


def get_default_filter_conveyor() -> ChunkFilterConveyor:
    """Returns singleton instance of the default chunk filter conveyor."""
    global _DEFAULT_CONVEYOR
    if _DEFAULT_CONVEYOR is None:
        _DEFAULT_CONVEYOR = build_chunk_filter_conveyor()
    return _DEFAULT_CONVEYOR
