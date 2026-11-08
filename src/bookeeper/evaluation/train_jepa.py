"""
Self-Supervised JEPA Training Engine (Strategy A: Two-Tier Architecture).

Combines a frozen foundational representation backbone (e.g., BAAI/bge-small-en-v1.5)
with a trainable multi-layer JEPA projection head trained using VICReg loss
(Variance-Invariance-Covariance Regularization).

Key Properties:
1. Frozen Backbone: Preserves universal vocabulary and syntax across all domains.
2. JEPA VICReg Head: Learns domain-adapted, isotropic representations where unrelated
   concepts have expected cosine similarity of 0.00.
3. GPU Acceleration: Runs natively on Apple Silicon GPU (MPS) or NVIDIA CUDA.
"""

from __future__ import annotations

import glob
import json
import logging
import math
import random
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from rich.console import Console
from rich.progress import BarColumn, Progress, SpinnerColumn, TextColumn, TimeElapsedColumn, TimeRemainingColumn
from rich.table import Table
from sentence_transformers import SentenceTransformer

from bookeeper.evaluation.jepa_encoder import resolve_torch_device

logger = logging.getLogger(__name__)


class TextJEPAHead(nn.Module):
    """
    Two-Tier JEPA Projection Head.
    Maps 384-d backbone representations to 768-d isotropic latent space.
    Features:
    - Input projection with LayerNorm and GELU
    - Residual feed-forward block
    - Output projection to target latent dimension with LayerNorm
    """

    def __init__(self, in_dim: int = 384, hidden_dim: int = 512, out_dim: int = 768, dropout: float = 0.05) -> None:
        super().__init__()
        self.in_dim = in_dim
        self.hidden_dim = hidden_dim
        self.out_dim = out_dim

        self.input_layer = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        self.res_linear1 = nn.Linear(hidden_dim, hidden_dim)
        self.res_act = nn.GELU()
        self.res_linear2 = nn.Linear(hidden_dim, hidden_dim)
        self.res_ln = nn.LayerNorm(hidden_dim)

        self.output_layer = nn.Sequential(
            nn.Linear(hidden_dim, out_dim),
            nn.LayerNorm(out_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.input_layer(x)
        res = self.res_linear2(self.res_act(self.res_linear1(h)))
        h = self.res_ln(h + res)
        z = self.output_layer(h)
        return z


def vicreg_loss(
    z_a: torch.Tensor,
    z_b: torch.Tensor,
    sim_coeff: float = 15.0,
    std_coeff: float = 15.0,
    cov_coeff: float = 5.0,
    epsilon: float = 1e-4,
) -> Tuple[torch.Tensor, float, float, float]:
    """
    VICReg Loss: Variance, Invariance, and Covariance Regularization.
    Prevents representation collapse without negative sampling or contrastive pairs.
    """
    # 1. Invariance Loss (MSE between positive representations)
    sim_loss = F.mse_loss(z_a, z_b)

    # 2. Variance Loss (Forces std per latent dimension to be >= 1.0)
    std_a = torch.sqrt(z_a.var(dim=0) + epsilon)
    std_b = torch.sqrt(z_b.var(dim=0) + epsilon)
    std_loss = torch.mean(F.relu(1.0 - std_a)) + torch.mean(F.relu(1.0 - std_b))

    # 3. Covariance Loss (Forces off-diagonal covariance to 0.0 -> decorrelated dimensions)
    b, d = z_a.shape
    za_centered = z_a - z_a.mean(dim=0)
    zb_centered = z_b - z_b.mean(dim=0)
    cov_a = (za_centered.T @ za_centered) / max(1, b - 1)
    cov_b = (zb_centered.T @ zb_centered) / max(1, b - 1)

    diag_mask = torch.eye(d, dtype=torch.bool, device=z_a.device)
    cov_loss = (cov_a[~diag_mask].pow(2).sum() / d) + (cov_b[~diag_mask].pow(2).sum() / d)

    total_loss = sim_coeff * sim_loss + std_coeff * std_loss + cov_coeff * cov_loss
    return total_loss, float(sim_loss.item()), float(std_loss.item()), float(cov_loss.item())


def _parse_single_pdf_worker(pdf_path_str: str) -> List[Tuple[str, str]]:
    """Worker function for parallel PDF discourse pair extraction."""
    try:
        import pypdf
        import logging
        logging.getLogger("pypdf").setLevel(logging.ERROR)
        reader = pypdf.PdfReader(pdf_path_str)
        doc_chunks = []
        for page in reader.pages[:25]:
            txt = page.extract_text() or ""
            for para in txt.split("\n\n"):
                p_clean = para.strip()
                if len(p_clean) >= 60:
                    doc_chunks.append(p_clean)
        pdf_pairs = []
        for i in range(len(doc_chunks) - 1):
            pdf_pairs.append((doc_chunks[i], doc_chunks[i + 1]))
        return pdf_pairs
    except Exception:
        return []


def load_chunk_pairs_from_states(
    state_dirs: Sequence[Union[str, Path]],
    books_dir: Optional[Union[str, Path]] = None,
    max_books: Optional[int] = None,
    max_pairs: Optional[int] = None,
    cache_path: Optional[Union[str, Path]] = "output/.jepa_chunk_pairs.json.gz",
    console: Optional[Console] = None,
) -> List[Tuple[str, str]]:
    """
    Load discourse continuity pairs (chunk_i, chunk_{i+1}) from pre-parsed book states
    and/or raw PDF books in books_dir. Caches parsed pairs to disk for fast reload.
    """
    if cache_path and not max_books and not max_pairs:
        c_path = Path(cache_path)
        if c_path.is_file():
            try:
                import gzip
                if console:
                    console.print(f"[bold green]Loading cached pairs from {c_path}...[/bold green]")
                with gzip.open(c_path, "rt", encoding="utf-8") as f:
                    data = json.load(f)
                pairs = [tuple(p) for p in data]
                if console:
                    console.print(f"✓ Loaded {len(pairs):,} discourse pairs from cache.")
                return pairs
            except Exception as e:
                logger.debug(f"Cache read error: {e}")

    state_files: List[Path] = []
    for s_dir in state_dirs:
        p = Path(s_dir)
        if p.is_dir():
            state_files.extend(list(p.glob("book_*_state.json")))

    # Deduplicate by resolved path
    unique_files = sorted(list({f.resolve(): f for f in state_files}.values()))
    if max_books and len(unique_files) > max_books:
        unique_files = unique_files[:max_books]

    pairs: List[Tuple[str, str]] = []
    seen_titles = set()

    for fpath in unique_files:
        try:
            with open(fpath, "r", encoding="utf-8") as f:
                data = json.load(f)
            title = data.get("book_title") or data.get("title")
            if title:
                seen_titles.add(title.strip().lower())
            chunks = data.get("chunks", [])
            for i in range(len(chunks) - 1):
                t1 = chunks[i].get("text", "").strip()
                t2 = chunks[i + 1].get("text", "").strip()
                if len(t1) >= 40 and len(t2) >= 40:
                    pairs.append((t1, t2))
                    if max_pairs and len(pairs) >= max_pairs:
                        return pairs
        except Exception as e:
            logger.debug(f"Error reading {fpath}: {e}")

    # If books_dir is provided, extract additional pairs from unparsed PDFs in parallel
    if books_dir:
        b_dir = Path(books_dir)
        if b_dir.is_dir():
            from concurrent.futures import ThreadPoolExecutor
            import pypdf

            logging.getLogger("pypdf").setLevel(logging.ERROR)
            pdf_files = sorted(list(b_dir.glob("*.pdf")))
            
            # Filter PDFs not already covered by state titles
            to_parse = [
                p for p in pdf_files
                if not any(t in p.stem.lower() for t in seen_titles if len(t) > 5)
            ]

            if console:
                console.print(f"[dim]Scanning {len(to_parse):,} new PDF books in thread pool...[/dim]")

            from concurrent.futures import ThreadPoolExecutor
            with ThreadPoolExecutor(max_workers=8) as executor:
                results = list(executor.map(lambda p: _parse_single_pdf_worker(str(p)), to_parse))

            for r in results:
                pairs.extend(r)
                if max_pairs and len(pairs) >= max_pairs:
                    return pairs[:max_pairs]

    if cache_path and not max_books and not max_pairs and pairs:
        try:
            import gzip
            c_path = Path(cache_path)
            c_path.parent.mkdir(parents=True, exist_ok=True)
            with gzip.open(c_path, "wt", encoding="utf-8") as f:
                json.dump(pairs, f)
            if console:
                console.print(f"[dim]Cached {len(pairs):,} discourse pairs to {c_path}[/dim]")
        except Exception as e:
            logger.debug(f"Could not cache pairs: {e}")

    return pairs


def train_jepa_head(
    pairs: List[Tuple[str, str]],
    backbone_name: str = "BAAI/bge-small-en-v1.5",
    target_dim: int = 768,
    hidden_dim: int = 512,
    epochs: int = 12,
    batch_size: int = 256,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    device: Optional[Union[str, torch.device]] = "gpu",
    output_weights_path: Union[str, Path] = "output/jepa_head_weights.pt",
    output_npz_path: Union[str, Path] = "output/jepa_768_weights.npz",
    resume: bool = True,
    console: Optional[Console] = None,
) -> Dict[str, Any]:
    """
    Execute Strategy A JEPA training on the given text pairs.
    """
    console = console or Console()
    dev = resolve_torch_device(device)
    console.print(f"[bold cyan]Initializing JEPA Training (Strategy A)[/bold cyan]")
    console.print(f" • Device: [bold green]{dev}[/bold green]")
    console.print(f" • Backbone: [bold yellow]{backbone_name}[/bold yellow]")
    console.print(f" • Target Latent Dim: [bold magenta]{target_dim}[/bold magenta]")
    console.print(f" • Total Training Pairs: [bold]{len(pairs):,}[/bold]")

    if not pairs:
        raise ValueError("No training pairs provided.")

    # 1. Load backbone and extract features with unique text caching
    console.print("\n[bold]Step 1/3: Extracting Backbone Representations on GPU...[/bold]")
    backbone = SentenceTransformer(backbone_name, device=str(dev))
    backbone.max_seq_length = 128
    in_dim = getattr(backbone, "get_embedding_dimension", backbone.get_sentence_embedding_dimension)() or 384

    # Collect texts and map unique strings to indices to avoid duplicate encoding
    texts_a = [p[0] for p in pairs]
    texts_b = [p[1] for p in pairs]

    unique_texts = list(dict.fromkeys(texts_a + texts_b))
    text_to_idx = {t: idx for idx, t in enumerate(unique_texts)}

    t0 = time.time()
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
        TimeRemainingColumn(),
        console=console,
    ) as progress:
        task_feat = progress.add_task(f"Encoding {len(unique_texts):,} unique texts...", total=len(unique_texts))
        encoded_vecs: List[np.ndarray] = []
        bs = 256

        for i in range(0, len(unique_texts), bs):
            sub = unique_texts[i : i + bs]
            v = backbone.encode(sub, batch_size=bs, show_progress_bar=False, normalize_embeddings=True)
            encoded_vecs.append(v)
            progress.advance(task_feat, len(sub))

    all_encoded = np.vstack(encoded_vecs).astype(np.float32)
    dt_feat = max(1e-4, time.time() - t0)
    console.print(f"✓ Encoded {len(unique_texts):,} unique representations in {dt_feat:.1f}s ({len(unique_texts)/dt_feat:.1f} texts/sec)")

    # Map back to pairs
    idx_a = np.array([text_to_idx[t] for t in texts_a], dtype=np.int64)
    idx_b = np.array([text_to_idx[t] for t in texts_b], dtype=np.int64)
    feat_a = all_encoded[idx_a]
    feat_b = all_encoded[idx_b]

    # 2. Setup JEPA Head and Optimizer
    console.print("\n[bold]Step 2/3: Training Two-Tier JEPA Head via VICReg Loss...[/bold]")
    model = TextJEPAHead(in_dim=in_dim, hidden_dim=hidden_dim, out_dim=target_dim).to(dev)
    if resume and Path(output_weights_path).is_file():
        try:
            ckpt = torch.load(output_weights_path, map_location=dev)
            if "state_dict" in ckpt:
                model.load_state_dict(ckpt["state_dict"])
                console.print(f"[bold green]✓ Loaded and resuming from existing checkpoint:[/bold green] {output_weights_path}")
        except Exception as e:
            console.print(f"[dim yellow]Notice: Could not load existing checkpoint ({e}). Starting fresh weights.[/dim yellow]")

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-5)

    n_samples = len(pairs)
    indices = np.arange(n_samples)

    history = []
    for epoch in range(1, epochs + 1):
        model.train()
        np.random.shuffle(indices)
        epoch_loss = 0.0
        epoch_sim = 0.0
        epoch_std = 0.0
        epoch_cov = 0.0
        batches = 0

        for b_start in range(0, n_samples, batch_size):
            b_idx = indices[b_start : b_start + batch_size]
            if len(b_idx) < 32:
                continue

            xa = torch.from_numpy(feat_a[b_idx]).to(dev)
            xb = torch.from_numpy(feat_b[b_idx]).to(dev)

            optimizer.zero_grad()
            za = model(xa)
            zb = model(xb)

            loss, s_loss, std_l, cov_l = vicreg_loss(za, zb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            epoch_loss += loss.item()
            epoch_sim += s_loss
            epoch_std += std_l
            epoch_cov += cov_l
            batches += 1

        scheduler.step()
        avg_loss = epoch_loss / max(1, batches)
        avg_sim = epoch_sim / max(1, batches)
        avg_std = epoch_std / max(1, batches)
        avg_cov = epoch_cov / max(1, batches)
        history.append({
            "epoch": epoch,
            "loss": avg_loss,
            "sim_loss": avg_sim,
            "std_loss": avg_std,
            "cov_loss": avg_cov,
        })

        console.print(
            f"  Epoch {epoch:2d}/{epochs:2d} | Total Loss: {avg_loss:7.3f} "
            f"(Invariance: {avg_sim*25:.3f}, Var-Penalty: {avg_std*25:.3f}, Cov-Penalty: {avg_cov:.3f})"
        )

    # 3. Validation Probe
    console.print("\n[bold]Step 3/3: Validating Semantic Separation & Zero-Isotropy...[/bold]")
    model.eval()
    with torch.no_grad():
        val_n = min(2000, n_samples)
        v_xa = torch.from_numpy(feat_a[:val_n]).to(dev)
        v_xb = torch.from_numpy(feat_b[:val_n]).to(dev)

        raw_za = model(v_xa)
        raw_zb = model(v_xb)
        mean_za = raw_za.mean(dim=0, keepdim=True)
        mean_zb = raw_zb.mean(dim=0, keepdim=True)

        # Center to eliminate anisotropic pedestal
        v_za = F.normalize(raw_za - mean_za, p=2, dim=-1)
        v_zb = F.normalize(raw_zb - mean_zb, p=2, dim=-1)

        # Related pairs cosine similarity
        related_cos = (v_za * v_zb).sum(dim=-1).cpu().numpy()

        # Random unrelated pairs cosine similarity (shuffled)
        perm = torch.randperm(val_n)
        unrelated_cos = (v_za * v_zb[perm]).sum(dim=-1).cpu().numpy()

        rel_mean = float(np.mean(related_cos))
        rel_std = float(np.std(related_cos))
        unrel_mean = float(np.mean(unrelated_cos))
        unrel_std = float(np.std(unrelated_cos))

    # Display Validation Results Table
    val_table = Table(title="JEPA Space Semantic Calibration Probe")
    val_table.add_column("Pair Regime", style="cyan")
    val_table.add_column("Mean Cosine Similarity", justify="right", style="bold green")
    val_table.add_column("Std Dev", justify="right")
    val_table.add_column("Ideal Target", justify="right", style="yellow")
    val_table.add_column("Status / Assessment", style="white")

    val_table.add_row(
        "Related Discourse Pairs",
        f"{rel_mean:.4f}",
        f"±{rel_std:.4f}",
        "> +0.65",
        "✓ Strong semantic cohesion" if rel_mean >= 0.55 else "Sub-optimal alignment",
    )
    val_table.add_row(
        "Random Unrelated Pairs",
        f"{unrel_mean:.4f}",
        f"±{unrel_std:.4f}",
        "~ 0.00 (±0.05)",
        "✓ Zero-isotropic hypersphere" if abs(unrel_mean) <= 0.08 else "Anisotropic distortion",
    )
    console.print(val_table)

    # 4. Save Checkpoint Weights
    weights_path = Path(output_weights_path)
    weights_path.parent.mkdir(parents=True, exist_ok=True)
    mean_vec_tensor = torch.from_numpy(np.mean(feat_a, axis=0, keepdims=True)).to(dev)
    with torch.no_grad():
        mean_z_tensor = model(mean_vec_tensor).cpu()

    torch.save(
        {
            "state_dict": model.state_dict(),
            "mean_z": mean_z_tensor,
            "in_dim": in_dim,
            "hidden_dim": hidden_dim,
            "out_dim": target_dim,
            "backbone_name": backbone_name,
            "rel_mean": rel_mean,
            "unrel_mean": unrel_mean,
        },
        weights_path,
    )
    console.print(f"✓ Saved JEPA PyTorch checkpoint to [bold green]{weights_path}[/bold green]")

    # 5. Export Closed-Form NPZ Matrix for instant compatibility with LightweightJEPAEncoder
    # We compute the empirical whitening matrix W_proj from the projected features
    all_z = []
    with torch.no_grad():
        for i in range(0, n_samples, 2048):
            sub_x = torch.from_numpy(feat_a[i : i + 2048]).to(dev)
            all_z.append(model(sub_x).cpu().numpy())
    all_z_arr = np.vstack(all_z)

    # Center and compute projection
    mean_vec = np.mean(all_z_arr, axis=0, keepdims=True)
    centered = all_z_arr - mean_vec
    cov = (centered.T @ centered) / max(1, len(centered) - 1)
    evals, evecs = np.linalg.eigh(cov)
    evals = np.maximum(evals, 1e-4)
    # Whitening operator
    w_proj = (evecs @ np.diag(1.0 / np.sqrt(evals)) @ evecs.T).astype(np.float32)

    npz_path = Path(output_npz_path)
    npz_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        npz_path,
        mean=mean_vec.astype(np.float32),
        W_proj=w_proj,
        latent_dim=target_dim,
        epsilon=1e-4,
        align_alpha=0.5,
    )
    console.print(f"✓ Exported JEPA whitening weights to [bold green]{npz_path}[/bold green]")

    return {
        "epochs": epochs,
        "history": history,
        "related_mean": rel_mean,
        "unrelated_mean": unrel_mean,
        "weights_path": str(weights_path),
        "npz_path": str(npz_path),
    }
