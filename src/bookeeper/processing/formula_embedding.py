"""
Hybrid formula-aware composite vector embedding for Knowledge Graph verification.
Implements Option A (Block-Normalized Composite Vector) enabling standard SIMD cosine similarity
across hybrid text and mathematical formula representations.
"""

import logging
import re
import zlib
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np

from bookeeper.processing.extractor import ExtractedFormula, extract_raw_formulas_from_text
from bookeeper.processing.filters import has_mathematical_formula

logger = logging.getLogger(__name__)

# ==============================================================================
# Math Primitives, Operators, AST Transitions, and Bilingual Lexicons
# ==============================================================================

# Dimensions 0..31: 32 Canonical Structural Operators
CANONICAL_OPERATORS = [
    "=",           # 0: Equality / assignment
    r"\sum",       # 1: Summation
    r"\int",       # 2: Integral
    r"\prod",      # 3: Product
    r"\frac",      # 4: Fraction / ratio
    r"\nabla",     # 5: Gradient
    r"\mathbb{E}", # 6: Expectation
    r"\mathcal{L}",# 7: Loss
    r"\arg\max",   # 8: Argmax
    r"\argmin",    # 9: Argmin
    r"\log",       # 10: Logarithm
    r"\exp",       # 11: Exponential
    r"\sim",       # 12: Distributed as (~, \sim)
    r"\in",        # 13: Element of
    r"\notin",     # 14: Not in
    r"^T",         # 15: Transpose
    r"\|",         # 16: Norm (||)
    r"\le",        # 17: Less or equal (<=, \le, ≤)
    r"\ge",        # 18: Greater or equal (>=, \ge, ≥)
    r"\partial",   # 19: Partial derivative
    r"\sqrt",      # 20: Square root
    r"\approx",    # 21: Approximately equal
    r"\equiv",     # 22: Equivalent
    r"\forall",    # 23: For all
    r"\exists",    # 24: Exists
    r"P(",         # 25: Probability function
    r"KL",         # 26: KL divergence
    r"\cdot",      # 27: Dot product / scalar multiplication
    r"\pm",        # 28: Plus-minus
    r"\infty",     # 29: Infinity
    r"\det",       # 30: Determinant
    r"\Delta",     # 31: Difference / Laplacian
]

OPERATOR_IDF_WEIGHTS: Dict[str, float] = {
    "=": 1.0,
    r"\sum": 1.8,
    r"\int": 2.0,
    r"\prod": 2.0,
    r"\frac": 1.5,
    r"\nabla": 2.5,
    r"\mathbb{E}": 2.2,
    r"\mathcal{L}": 2.0,
    r"\arg\max": 2.5,
    r"\argmin": 2.5,
    r"\log": 1.5,
    r"\exp": 1.5,
    r"\sim": 1.8,
    r"\in": 1.2,
    r"\notin": 2.0,
    r"^T": 1.8,
    r"\|": 1.8,
    r"\le": 1.2,
    r"\ge": 1.2,
    r"\partial": 2.2,
    r"\sqrt": 1.8,
    r"\approx": 1.5,
    r"\equiv": 2.0,
    r"\forall": 1.8,
    r"\exists": 2.0,
    r"P(": 1.8,
    r"KL": 2.5,
    r"\cdot": 1.2,
    r"\pm": 1.5,
    r"\infty": 1.5,
    r"\det": 2.2,
    r"\Delta": 1.8,
}

# Dimensions 64..127: Normalized Variable & Domain Identifiers
VARIABLES_LIST = [
    r"\theta", r"\phi", r"\pi", r"\alpha", r"\beta", r"\gamma", r"\delta",
    r"\epsilon", r"\lambda", r"\mu", r"\sigma", r"\tau", r"\omega",
    "KL", "Q", "V", "W", "b", "x", "y", "z", "s", "a", "r", "g",
    r"\mathcal{D}", "st", "st+k", "st:t+k", "st+1", "at", "rt", "yt", "xt",
    "st_k", "at_1", "phi_omega", "phi", "omega", "theta_pi"
]

def _stable_hash(key: str) -> int:
    """Deterministic CRC32 hash for consistent bucket assignment across processes."""
    return zlib.crc32(key.encode("utf-8"))

# Bilingual Mapping (English text -> LaTeX structural & variable buckets)
ENGLISH_MATH_KEYWORD_MAP: Dict[str, List[int]] = {
    "expectation": [6],
    "expected value": [6],
    "expected": [6],
    "gradient": [5],
    "policy gradient": [5, 64 + (_stable_hash(r"\pi") % 64)],
    "derivative": [5, 19],
    "partial derivative": [19],
    "loss": [7],
    "objective": [7],
    "loss function": [7],
    "cost function": [7],
    "kl divergence": [26, 64 + (_stable_hash("KL") % 64)],
    "relative entropy": [26, 64 + (_stable_hash("KL") % 64)],
    "summation": [1],
    "sum": [1],
    "probability": [25],
    "likelihood": [25, 10],
    "integral": [2],
    "integrate": [2],
    "logarithm": [10],
    "log likelihood": [10, 25],
    "log-likelihood": [10, 25],
    "exponential": [11],
    "softmax": [11, 1],
    "variance": [64 + (_stable_hash(r"\sigma") % 64)],
    "covariance": [64 + (_stable_hash(r"\sigma") % 64)],
    "argmax": [8],
    "argmin": [9],
    "bellman": [64 + (_stable_hash("Q") % 64), 64 + (_stable_hash("V") % 64)],
    "temporal difference": [64 + (_stable_hash("Q") % 64), 64 + (_stable_hash("V") % 64)],
    "td loss": [7, 64 + (_stable_hash("Q") % 64)],
    "td error": [64 + (_stable_hash("Q") % 64)],
    "q-learning": [64 + (_stable_hash("Q") % 64)],
    "policy": [64 + (_stable_hash(r"\pi") % 64)],
    "parameters": [64 + (_stable_hash(r"\theta") % 64)],
    "weights": [64 + (_stable_hash("W") % 64)],
    "norm": [16],
    "l2 norm": [16],
    "transpose": [15],
    "square root": [20],
    "subgoal": [64 + (_stable_hash("g") % 64)],
    "discount factor": [64 + (_stable_hash(r"\gamma") % 64)],
    "gamma": [64 + (_stable_hash(r"\gamma") % 64)],
    "theta": [64 + (_stable_hash(r"\theta") % 64)],
    "phi": [64 + (_stable_hash(r"\phi") % 64)],
    "alpha": [64 + (_stable_hash(r"\alpha") % 64)],
    "beta": [64 + (_stable_hash(r"\beta") % 64)],
    "lambda": [64 + (_stable_hash(r"\lambda") % 64)],
    "epsilon": [64 + (_stable_hash(r"\epsilon") % 64)],
    "sigma": [64 + (_stable_hash(r"\sigma") % 64)],
    "replay buffer": [64 + (_stable_hash(r"\mathcal{D}") % 64)],
    "determinant": [30],
    "laplacian": [31],
}


def build_formula_feature_vector(text_or_formulas: Union[str, Sequence[Any]], m: int = 128) -> np.ndarray:
    """
    Constructs a fixed-size mathematical feature vector v_math in R^m (default m=128)
    using TF-IDF operator counts, 2-gram AST transition hashing, and variable identifiers.

    Dimensions:
      0..31:  Canonical structural operators (weighted by IDF)
      32..63: 2-gram structural transitions / AST layout paths (hashed into 32 buckets)
      64..127: Normalized variables and domain identifiers (hashed into 64 buckets)
      Bilingual mapping: English math terms map to exact same operator/variable indices.
    """
    vec = np.zeros(m, dtype=np.float32)

    if isinstance(text_or_formulas, str):
        full_text = text_or_formulas
    elif isinstance(text_or_formulas, (list, tuple, set)):
        items: List[str] = []
        for x in text_or_formulas:
            if isinstance(x, str):
                items.append(x)
            elif isinstance(x, dict):
                items.append(str(x.get("raw_latex") or x.get("latex") or x.get("text") or ""))
            elif hasattr(x, "raw_latex"):
                items.append(str(getattr(x, "raw_latex", "")))
            elif hasattr(x, "text"):
                items.append(str(getattr(x, "text", "")))
            else:
                items.append(str(x))
        full_text = " ".join(items)
    else:
        full_text = str(text_or_formulas)

    if not full_text.strip():
        return vec

    # 1. Dimensions 0..31: Structural Operators & Patterns
    for idx, op in enumerate(CANONICAL_OPERATORS):
        pattern = re.escape(op)
        if op == "=":
            pattern = r"(?<![=!<>])=(?!=)"
        elif op == r"\|":
            pattern = r"\|(?:\||\s*\|)?"
        elif op == "P(":
            pattern = r"\bP\s*\("
        elif op == "KL":
            pattern = r"\bKL\b"

        matches = len(re.findall(pattern, full_text, re.IGNORECASE if op in ("KL", "P(") else 0))
        if matches > 0:
            weight = OPERATOR_IDF_WEIGHTS.get(op, 1.0)
            vec[idx] += float(matches * weight)

    # 2. Dimensions 32..63: 2-gram Structural Transitions (AST Layout Paths)
    op_positions: List[Tuple[int, str]] = []
    for op in CANONICAL_OPERATORS:
        pattern = re.escape(op)
        if op == "=":
            pattern = r"(?<![=!<>])=(?!=)"
        elif op == r"\|":
            pattern = r"\|(?:\||\s*\|)?"
        elif op == "P(":
            pattern = r"\bP\s*\("
        elif op == "KL":
            pattern = r"\bKL\b"

        for m_obj in re.finditer(pattern, full_text, re.IGNORECASE if op in ("KL", "P(") else 0):
            op_positions.append((m_obj.start(), op))

    op_positions.sort(key=lambda x: x[0])
    ordered_ops = [op for _, op in op_positions]

    for i in range(len(ordered_ops) - 1):
        pair = f"{ordered_ops[i]}->{ordered_ops[i+1]}"
        bucket = 32 + (_stable_hash(pair) % 32)
        vec[bucket] += 1.5

    # 3. Dimensions 64..127: Normalized Variable & Domain Identifiers
    for var in VARIABLES_LIST:
        if len(var) == 1 and var.isalpha():
            var_pat = r"\b" + var + r"\b"
        elif not var.startswith("\\"):
            var_pat = r"\b" + re.escape(var) + r"\b"
        else:
            var_pat = re.escape(var)

        matches = len(re.findall(var_pat, full_text))
        if matches > 0:
            bucket = 64 + (_stable_hash(var) % 64)
            vec[bucket] += float(matches * 1.0)

    # 4. Bilingual Mapping: English keywords to corresponding indices
    lower_text = full_text.lower()
    for kw, target_indices in ENGLISH_MATH_KEYWORD_MAP.items():
        if re.search(r"\b" + re.escape(kw) + r"\b", lower_text):
            for t_idx in target_indices:
                if t_idx < m:
                    vec[t_idx] += 1.8

    return vec


def build_composite_embedding(
    text_vec: np.ndarray,
    math_vec: Optional[np.ndarray] = None,
    has_formula: bool = False,
    alpha: float = 0.35,
    m: int = 128,
) -> np.ndarray:
    """
    Constructs a block-normalized composite unit vector z in R^{d + m}:
      z = [ sqrt(1 - alpha) * v_hat_text ; sqrt(alpha) * v_hat_math ]  if has_formula
      z = [ v_hat_text ; 0_m ]                                        otherwise

    Guarantees ||z||_2 = 1.0 strictly, allowing single standard cosine dot product.
    """
    v_text = np.asarray(text_vec, dtype=np.float32).flatten()
    norm_t = float(np.linalg.norm(v_text))
    v_text_hat = v_text / max(norm_t, 1e-12)

    # If math vector not explicitly provided, create zero vector of length m
    if math_vec is None:
        v_math = np.zeros(m, dtype=np.float32)
    else:
        v_math = np.asarray(math_vec, dtype=np.float32).flatten()
        if len(v_math) < m:
            v_math = np.pad(v_math, (0, m - len(v_math)))
        elif len(v_math) > m:
            v_math = v_math[:m]

    norm_m = float(np.linalg.norm(v_math))

    # Apply hybrid scaling if has_formula and non-zero math vector
    if has_formula and norm_m > 1e-12 and alpha > 0.0:
        eff_alpha = min(1.0, max(0.0, float(alpha)))
        v_math_hat = v_math / norm_m
        scaled_text = np.sqrt(1.0 - eff_alpha, dtype=np.float32) * v_text_hat
        scaled_math = np.sqrt(eff_alpha, dtype=np.float32) * v_math_hat
        z = np.concatenate([scaled_text, scaled_math])
    else:
        z = np.concatenate([v_text_hat, np.zeros(m, dtype=np.float32)])

    # Ensure numerical unit length
    final_norm = float(np.linalg.norm(z))
    return z / max(final_norm, 1e-12)


def build_composite_embeddings_matrix(
    text_matrix: np.ndarray,
    has_formula_mask: np.ndarray,
    math_vectors: Optional[np.ndarray] = None,
    text_sources: Optional[Sequence[Any]] = None,
    alpha: float = 0.35,
    m: int = 128,
) -> np.ndarray:
    """
    Batch conversion of text matrix (N, d) into block-normalized composite matrix (N, d + m).
    """
    if len(text_matrix) == 0:
        d = text_matrix.shape[1] if text_matrix.ndim > 1 else 0
        return np.empty((0, d + m), dtype=np.float32)

    n_items = len(text_matrix)
    d = text_matrix.shape[1]
    composite_matrix = np.empty((n_items, d + m), dtype=np.float32)

    # Pre-build math vectors if text sources provided
    for i in range(n_items):
        t_vec = text_matrix[i]
        has_f = bool(has_formula_mask[i]) if i < len(has_formula_mask) else False
        m_vec: Optional[np.ndarray] = None
        if math_vectors is not None and i < len(math_vectors):
            m_vec = math_vectors[i]
        elif has_f and text_sources is not None and i < len(text_sources):
            m_vec = build_formula_feature_vector(text_sources[i], m=m)

        composite_matrix[i] = build_composite_embedding(
            text_vec=t_vec,
            math_vec=m_vec,
            has_formula=has_f,
            alpha=alpha,
            m=m,
        )

    return composite_matrix


def compute_adaptive_cosine(
    z1: np.ndarray,
    z2: np.ndarray,
    formula_dim: int = 128,
    alpha: float = 0.35,
) -> float:
    """
    Computes unpenalized adaptive cosine similarity between two composite vectors:
      z = [ v_text ; v_math ]

    Guarantees:
    - If either vector has no formula (all-zero math subspace), returns pure text cosine similarity,
      completely eliminating the artificial sqrt(1 - alpha) penalty.
    - If both vectors have active formulas, returns max(text_cosine, (1 - alpha) * text_cosine + alpha * math_cosine),
      ensuring mathematical structure boosts matches without ever degrading below text alignment.
    """
    v1 = np.asarray(z1, dtype=np.float32).flatten()
    v2 = np.asarray(z2, dtype=np.float32).flatten()

    if len(v1) == 0 or len(v2) == 0:
        return 0.0

    if len(v1) != len(v2) or len(v1) <= formula_dim or formula_dim <= 0:
        norm1 = float(np.linalg.norm(v1))
        norm2 = float(np.linalg.norm(v2))
        return float(np.dot(v1, v2) / (max(norm1, 1e-12) * max(norm2, 1e-12)))

    d = len(v1) - formula_dim
    t1, m1 = v1[:d], v1[d:]
    t2, m2 = v2[:d], v2[d:]

    norm_t1 = float(np.linalg.norm(t1))
    norm_t2 = float(np.linalg.norm(t2))
    sim_t = float(np.dot(t1, t2) / (max(norm_t1, 1e-12) * max(norm_t2, 1e-12)))

    norm_m1 = float(np.linalg.norm(m1))
    norm_m2 = float(np.linalg.norm(m2))

    if norm_m1 > 1e-6 and norm_m2 > 1e-6:
        sim_m = float(np.dot(m1, m2) / (norm_m1 * norm_m2))
        eff_alpha = min(1.0, max(0.0, float(alpha)))
        sim_h = (1.0 - eff_alpha) * sim_t + eff_alpha * sim_m
        return float(max(sim_t, sim_h))

    return sim_t
