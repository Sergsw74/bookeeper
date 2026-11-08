# 🎓 Mandatory Domain JEPA Pre-Training & Calibration Guide

> [!CAUTION]
> **PRE-TRAINING IS MANDATORY BEFORE BUILDING KNOWLEDGE GRAPHS**:  
> In `bookeeper`, the 768-d JEPA representation pipeline uses a **Two-Tier Architecture**: a frozen sentence backbone (`bge-small-en-v1.5`) coupled with a trainable multi-layer projection head (`TextJEPAHead`).  
> **If you run `bookeeper build-graph`, `sweep-thresholds`, or semantic search without pre-training the projection head, the linear layers remain PyTorch-initialized random matrices!**  
> Running on random weights causes **representation collapse and chaotic cosine similarities**, producing arbitrary filtering decisions and broken concept matching. You **must** train or supply calibrated weights (`output/jepa_head_weights.pt` and `output/jepa_768_weights.npz`) before production ingestion.

---

## 🔬 Why Pre-Training is Mandatory: The Anisotropy & Random Weight Problem

### 1. The Untrained Base Model / Random Head Failure
When initialized without trained weights:
- `TextJEPAHead` initializes weights with PyTorch default uniform random linear layers.
- Every text chunk passing through the head undergoes random matrix multiplication, destroying semantic proximity and producing arbitrary, erratic vector directions.
- Positive discourse pairs (e.g. consecutive paragraphs in a proof) receive random cosine similarities ranging from $-0.30$ to $+0.30$, causing identical concepts to be evaluated as novel and triggering thousands of redundant, costly LLM inference calls.

### 2. The Classical Base Model Anisotropy Problem (The "Cone Effect")
Standard off-the-shelf sentence transformers (e.g. raw BGE-M3 or BGE-small without whitening):
- Suffer from severe **representation anisotropy**: embeddings are confined to a narrow directional cone in hyperspace.
- **Random, completely unrelated text chunks average ~0.55 cosine similarity**!
- Because baseline noise occupies $[0.50, 0.60]$, the usable dynamic range is compressed to a tiny band $[0.55, 0.95]$. A threshold shift of $0.05$ causes violent swings between over-filtering and total pipeline clogging.

### 3. The Calibrated JEPA Solution
`bookeeper train-jepa` solves both problems via **VICReg (Variance-Invariance-Covariance Regularization)** self-supervised training on local literature discourse pairs:

```mermaid
flowchart LR
    subgraph RawBackbone ["1. Frozen Representation Backbone"]
        T1["Chunk A (Context)"] --> B1["BAAI/bge-small-en-v1.5\n(Frozen 33M params, 384-d)"]
        T2["Chunk B (Discourse Follow-up)"] --> B2["BAAI/bge-small-en-v1.5\n(Frozen 33M params, 384-d)"]
    end

    subgraph TrainableHead ["2. Two-Tier Projection Head (Trainable)"]
        B1 --> H1["TextJEPAHead (0.65M params)\nLayerNorm -> GELU -> Residual MLP -> 768-d"]
        B2 --> H2["TextJEPAHead (0.65M params)\nLayerNorm -> GELU -> Residual MLP -> 768-d"]
    end

    subgraph VICReg ["3. VICReg Loss Objective"]
        H1 --> Z1["z_A (768-d)"]
        H2 --> Z2["z_B (768-d)"]
        Z1 & Z2 --> Loss{"VICReg Loss\n15.0 * Invariance\n+ 15.0 * Variance (std >= 1.0)\n+ 5.0 * Covariance (off-diag = 0)"}
    end
```

- **Invariance Criterion ($\lambda = 15.0$)**: Forces adjacent chunks within the same section/chapter to map close together ($>0.65$).
- **Variance Regularization ($\mu = 15.0$)**: Forces the standard deviation along every latent dimension to be $\ge 1.0$, completely preventing representation collapse.
- **Covariance Decorrelation ($\nu = 5.0$)**: Drives off-diagonal covariance between different latent dimensions to strictly $0.00$, making dimensions mutually orthogonal.
- **Zero-Centered Isotropic Baseline**: Unrelated concepts produce an expected cosine similarity of strictly **$\mu = -0.008 \pm 0.044$** (effectively **$0.00$**), unlocking the full $[-0.05, +0.95]$ dynamic range!

---

## 🛠️ CLI Command Reference: `bookeeper train-jepa`

`bookeeper` provides a dedicated, high-speed CLI training command:

```bash
bookeeper train-jepa [OPTIONS]
```

### Options & Parameters

| Option | Shorthand | Default | Description |
| :--- | :---: | :---: | :--- |
| `--state-dir` | `-s` | `output/.book_states` | Directory containing parsed `book_*_state.json` files to harvest discourse pairs. |
| `--books-dir` | `--download-dir` | `None` | Optional directory containing raw PDF whitepapers or Calibre books to teach from. |
| `--epochs` | `-e` | `10` | Number of training epochs (10–12 epochs recommended). |
| `--batch-size` | | `256` | Training mini-batch size (128–512 depending on VRAM/RAM). |
| `--lr` | | `0.001` | AdamW learning rate with Cosine Annealing schedule down to `1e-5`. |
| `--latent-dim` | `-d` | `768` | Target latent dimension (must be 768 for calibrated Honey Spot A filters). |
| `--device` | | `gpu` | Execution device: `gpu` (Apple Silicon MPS / NVIDIA CUDA), `mps`, `cuda`, or `cpu`. |
| `--output-weights` | | `output/jepa_head_weights.pt` | Output path for PyTorch checkpoint (`TextJEPAHead` weights + mean vector). |
| `--output-npz` | | `output/jepa_768_weights.npz` | Output path for closed-form whitening projection matrix. |
| `--resume / --no-resume` | | `True` | Automatically resumes and fine-tunes from existing checkpoint weights. |
| `--max-books` | `-b` | `None` | Optional cap on number of books to sample for rapid prototyping. |
| `--max-pairs` | `-p` | `None` | Optional cap on total chunk pairs for quick testing. |

---

## 🚀 Step-by-Step Training Instructions

### Workflow A: Teaching from Pre-Parsed Book States (Fastest)

If you have already processed or partially indexed books with `bookeeper build-graph`, discourse pairs are already segmented and ready in `.book_states/`:

```bash
# Pre-train / teach JEPA on GPU for 10 epochs:
bookeeper train-jepa \
  --state-dir output/.book_states \
  --epochs 10 \
  --batch-size 256 \
  --device gpu
```

**Execution Walkthrough**:
1. **Harvesting Pairs**: Scans `.book_states/` for adjacent chunks $(t_i, t_{i+1})$ where character length $\ge 40$. Caches pairs to `output/.jepa_chunk_pairs.json.gz` (17 MB).
2. **Step 1/3 (Backbone Encoding)**: Encodes all unique texts through frozen `bge-small-en-v1.5` on GPU (~1,800 texts/sec on Apple Silicon M4 GPU).
3. **Step 2/3 (VICReg Training)**: Trains the 0.65M parameter head for 10 epochs with Cosine Annealing learning rate (~1.5s per epoch).
4. **Step 3/3 (Zero-Isotropy Validation Probe)**: Automatically validates related vs unrelated pair separation.
5. **Artifact Generation**: Saves `output/jepa_head_weights.pt` (4.3 MB) and `output/jepa_768_weights.npz` (2.1 MB).

---

### Workflow B: Teaching Directly from Raw Academic PDFs

If you are bootstrapping a new library from scratch and have not run `build-graph` yet, you can teach the JEPA model directly from a folder of raw PDF papers:

```bash
# Scan download folder in parallel, extract discourse continuity, and train:
bookeeper train-jepa \
  --books-dir archive_downloader/downloads \
  --epochs 12 \
  --batch-size 256 \
  --device gpu
```

This extracts paragraph transitions across PDF pages in parallel thread pools, bypassing LLM extraction entirely.

---

## 📊 Empirical Benchmarks & Hardware Performance

Evaluated on the current production corpus (**314 books**, **18,458 literature chunks**, **18,000+ discourse continuity pairs**, 6.9M text characters):

### Execution Time: GPU vs CPU

| Pipeline Stage | Apple Silicon M4 GPU (`mps`) | 8-Core Intel/AMD CPU (`cpu`) | Hardware Acceleration |
| :--- | :---: | :---: | :---: |
| **Backbone Representation Extraction** | **10.2s** (~1,800 texts/s) | **68.4s** (~270 texts/s) | **6.7× Faster on GPU** |
| **VICReg Training Loop (10 Epochs)** | **15.6s** (1.56s / epoch) | **98.2s** (9.82s / epoch) | **6.3× Faster on GPU** |
| **Validation Probe & Whitening Matrix** | **1.1s** | **4.8s** | **4.4× Faster on GPU** |
| **Total End-to-End Training Duration** | **26.9 seconds** | **2.8 minutes** | **6.2× Overall Speedup** |

### Runtime Inference & Model Footprint

| Metric | Measured Specification | Comparison / Advantage |
| :--- | :---: | :--- |
| **Trained Head Checkpoint (`.pt`)** | **4.3 MB** | Ultra-compact, instant network transfer |
| **Whitening Matrix (`.npz`)** | **2.1 MB** | Microsecond NumPy/C-BLAS loading |
| **Single-Chunk Latency** | **0.054 ms** | **216× faster** than 567M transformer (28.6 ms) |
| **End-to-End Inference Throughput** | **>15,000 texts/sec** | Real-time stream processing on edge CPU/GPU |
| **Pairwise Similarity Speed** | **>5.1 Million pairs/sec** | Microsecond routing against 20,000+ graph concepts |
| **Idea-to-Chunk Similarity Speed** | **>14.6 Million pairs/sec** | Instant parallel attribution |

---

## 🔍 Validation Output: How to Interpret Training Results

Upon completion, `bookeeper train-jepa` prints the **Semantic Calibration Probe**:

```
                 JEPA Space Semantic Calibration Probe
┏━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━┳━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
┃ Pair Regime             ┃ Mean Cosine Similarity ┃  Std Dev ┃ Ideal Target ┃ Status / Assessment       ┃
┡━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━╇━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━━┩
│ Related Discourse Pairs │                +0.7506 │  ±0.1914 │      > +0.65 │ ✓ Strong semantic cohesion│
│ Random Unrelated Pairs  │                -0.0049 │  ±0.0572 │ ~ 0.00 (±0.05│ ✓ Zero-isotropic space    │
└─────────────────────────┴────────────────────────┴──────────┴──────────────┴───────────────────────────┘
```

### Pass / Fail Health Criteria:
- **`Related Discourse Pairs` $\ge +0.65$**: Confirms the head preserves continuity across narrative exposition and mathematical formulations.
- **`Random Unrelated Pairs` between $[-0.05, +0.05]$**: Confirms VICReg eliminated the anisotropic pedestal. If unrelated mean $> 0.15$, increase `--epochs` from 10 to 14 or increase covariance weight.
- **Standard Deviation $\le 0.06$**: Confirms the hypersphere is uniform without irregular clustering holes.

---

## ⚙️ Seamless Pipeline Integration

Once training completes, the resulting weights are automatically utilized across the entire `bookeeper` ecosystem:

1. **`config.yaml` Activation**:
   ```yaml
   embedding_model: "jepa-768"
   high_similarity_threshold: 0.85
   similarity_threshold: 0.65
   cover_similarity_threshold: 0.85
   dual_pass:
     similarity_threshold: 0.84
   ```
2. **`bookeeper build-graph`**: Automatically loads `output/jepa_head_weights.pt` and projects all candidate chunks in microseconds.
3. **`bookeeper sweep-thresholds`**: Automatically uses the 768-d JEPA vectors to simulate filtering variations and compute Chamfer coverage.
4. **Interactive Web UI**: Loads `output/jepa_768_weights.npz` to execute instant vector search and graph seed identification.

---

## ❓ Troubleshooting & FAQ

#### Q: What happens if `output/jepa_head_weights.pt` is missing?
The pipeline falls back to an untrained head with PyTorch default uniform random initialization. A warning will appear in the log:
```text
WARNING: JEPA head weights not found at output/jepa_head_weights.pt! The projection head is using uncalibrated random weights. Pre-training is mandatory for accurate cosine similarity: please run 'bookeeper train-jepa' first.
```
If you see this warning, run `bookeeper train-jepa` immediately to restore calibrated embeddings.

#### Q: How often should I retrain?
- **Small additions (1–20 books)**: No retraining needed. The frozen backbone generalizes zero-shot across new books.
- **Major new domain (e.g. adding 100 medicine textbooks to a computer science library)**: Run `bookeeper train-jepa --resume` for 5–8 epochs to adapt the representation manifold to the new vocabulary.

#### Q: Can I run this in Docker?
Yes! The Docker image includes PyTorch with CPU/CUDA support:
```bash
docker compose run --rm bookeeper train-jepa --epochs 10
```
