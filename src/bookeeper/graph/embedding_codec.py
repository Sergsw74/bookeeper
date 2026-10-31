"""
High-performance embedding compression and base64 binary encoding utilities.

Replaces verbose JSON float arrays (e.g., [-0.012345, 0.987654, ...]) with compact
packed IEEE binary buffers encoded in Base64 (float16 / float32).

Compression benchmarks for a 768-dimensional vector:
  - JSON float array: ~15.0 KB per embedding
  - Float32 Base64:   ~4.10 KB per embedding (~3.7x reduction)
  - Float16 Base64:   ~2.05 KB per embedding (~7.3x reduction, <1e-4 cosine error)
"""

import base64
import logging
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np

logger = logging.getLogger(__name__)

SUPPORTED_DTYPES = {
    "float16": np.float16,
    "float32": np.float32,
    "float64": np.float64,
}


def encode_embedding(
    vector: Union[List[float], Tuple[float, ...], np.ndarray],
    dtype: str = "float16",
) -> Dict[str, Any]:
    """
    Encode an embedding vector into a compact Base64 binary payload.

    Args:
        vector: 1D array or sequence of floating point numbers.
        dtype: Target binary precision ('float16' for ~7.3x compression, or 'float32' for ~3.7x).

    Returns:
        Dict containing:
          - "embedding_b64": Base64 string of packed raw binary bytes.
          - "embedding_dim": Length of the vector.
          - "embedding_dtype": Data type used for serialization ("float16" or "float32").
    """
    if vector is None:
        raise ValueError("Cannot encode None embedding vector")

    arr = np.asarray(vector, dtype=np.float32)
    if arr.ndim != 1:
        arr = arr.flatten()

    dim = int(arr.shape[0])
    if dim == 0:
        return {
            "embedding_b64": "",
            "embedding_dim": 0,
            "embedding_dtype": dtype,
        }

    target_np_type = SUPPORTED_DTYPES.get(dtype, np.float16)
    packed_bytes = arr.astype(target_np_type).tobytes()
    b64_str = base64.b64encode(packed_bytes).decode("ascii")

    return {
        "embedding_b64": b64_str,
        "embedding_dim": dim,
        "embedding_dtype": dtype,
    }


def decode_embedding(
    encoded: Union[str, Dict[str, Any]],
    dtype: Optional[str] = None,
    return_numpy: bool = True,
) -> Optional[Union[np.ndarray, List[float]]]:
    """
    Decode a Base64-encoded binary embedding into a float32 vector.

    Args:
        encoded: Either a Base64 string or a dict containing 'embedding_b64'
                 (and optional 'embedding_dtype').
        dtype: Explicit binary dtype ('float16' or 'float32'). If None, infers from dict
               or defaults to 'float16'.
        return_numpy: If True, returns np.ndarray (float32); if False, returns list of floats.

    Returns:
        Float32 array/list, or None if input was empty.
    """
    if not encoded:
        return None

    if isinstance(encoded, dict):
        b64_str = encoded.get("embedding_b64", "")
        eff_dtype = dtype or encoded.get("embedding_dtype", "float16")
    elif isinstance(encoded, str):
        b64_str = encoded
        eff_dtype = dtype or "float16"
    else:
        raise TypeError(f"Expected str or dict for decode_embedding, got {type(encoded)}")

    if not b64_str:
        return None

    raw_bytes = base64.b64decode(b64_str)
    np_dtype = SUPPORTED_DTYPES.get(eff_dtype, np.float16)
    arr = np.frombuffer(raw_bytes, dtype=np_dtype).astype(np.float32)

    return arr if return_numpy else arr.tolist()


def get_node_embedding(
    attrs: Dict[str, Any],
    return_numpy: bool = True,
) -> Optional[Union[np.ndarray, List[float]]]:
    """
    Universal accessor for a node's embedding vector.
    Transparently supports:
      1. Legacy raw float arrays in attrs["embedding"]
      2. Compact Base64 binary in attrs["embedding_b64"]
      3. Pre-parsed numpy arrays

    Args:
        attrs: Node attribute dictionary from NetworkX graph.
        return_numpy: If True, returns np.ndarray; if False, returns list of floats.

    Returns:
        Float32 embedding vector or None if missing.
    """
    if not attrs:
        return None

    # Case 1: Compact Base64 binary
    if "embedding_b64" in attrs and attrs["embedding_b64"]:
        dtype = attrs.get("embedding_dtype", "float16")
        return decode_embedding(attrs["embedding_b64"], dtype=dtype, return_numpy=return_numpy)

    # Case 2: Legacy float array / list
    if "embedding" in attrs and attrs["embedding"] is not None:
        emb = attrs["embedding"]
        if isinstance(emb, np.ndarray):
            arr = emb.astype(np.float32)
            return arr if return_numpy else arr.tolist()
        if isinstance(emb, (list, tuple)) and len(emb) > 0:
            if return_numpy:
                return np.asarray(emb, dtype=np.float32)
            return [float(x) for x in emb]

    return None


def compress_node_embedding(
    attrs: Dict[str, Any],
    dtype: str = "float16",
    in_place: bool = True,
) -> Dict[str, Any]:
    """
    Compress a node's 'embedding' list into 'embedding_b64' packed binary.
    Removes the verbose 'embedding' list from the dictionary to reclaim RAM/disk.

    Args:
        attrs: Node attributes dictionary.
        dtype: Target precision ('float16' recommended).
        in_place: Whether to modify attrs in-place or return a copy.

    Returns:
        Updated dictionary with 'embedding_b64', 'embedding_dim', and 'embedding_dtype'.
    """
    target = attrs if in_place else dict(attrs)

    # If already compressed, ensure clean state
    if "embedding_b64" in target and target["embedding_b64"]:
        target.pop("embedding", None)
        return target

    if "embedding" in target and target["embedding"] is not None:
        raw = target.pop("embedding")
        if isinstance(raw, (list, tuple, np.ndarray)) and len(raw) > 0:
            encoded = encode_embedding(raw, dtype=dtype)
            target.update(encoded)

    return target


def decompress_node_embedding(
    attrs: Dict[str, Any],
    as_list: bool = False,
    in_place: bool = True,
) -> Dict[str, Any]:
    """
    Decompress a node's 'embedding_b64' into 'embedding'.

    Args:
        attrs: Node attributes dictionary.
        as_list: If True, stores as list[float]; if False, stores as np.ndarray(float32).
        in_place: Whether to modify attrs in-place or return a copy.

    Returns:
        Updated dictionary with 'embedding' populated.
    """
    target = attrs if in_place else dict(attrs)

    if "embedding_b64" in target and target["embedding_b64"]:
        vec = decode_embedding(target, return_numpy=not as_list)
        if vec is not None:
            target["embedding"] = vec

    return target


def compare_vector_fidelity(
    original: Union[List[float], np.ndarray],
    decoded: Union[List[float], np.ndarray],
) -> Tuple[float, float]:
    """
    Compute cosine similarity and maximum absolute error between original and decoded vectors.

    Returns:
        (cosine_similarity, max_absolute_error)
    """
    v1 = np.asarray(original, dtype=np.float32)
    v2 = np.asarray(decoded, dtype=np.float32)

    norm1 = np.linalg.norm(v1)
    norm2 = np.linalg.norm(v2)
    if norm1 == 0.0 or norm2 == 0.0:
        cos_sim = 1.0 if norm1 == norm2 else 0.0
    else:
        cos_sim = float(np.dot(v1, v2) / (norm1 * norm2))

    max_err = float(np.max(np.abs(v1 - v2)))
    return cos_sim, max_err
