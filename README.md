# MERIT: Learning Disentangled Music Representations for Audio Similarity


MERIT learns three disentangled similarity representations: **melody**, **rhythm**, and **timbre** from a single frozen MERT-v1-330M backbone using contrastive learning with triplet data generated via the JASCO music generation model.

---

## Architecture

```
MERT-v1-330M (frozen)
  └─ Layers 3, 4, 5, 6, 23  →  mean-pool over time  →  concat  →  5120-dim

Per-factor MLP head (trained independently):
  Linear(5120 → 512, bias=True) → ReLU → Linear(512 → 128, bias=False) → L2-norm

Loss: Circle Loss (γ=10, m=0.2)
Optimizer: AdamW (lr=1e-3)
Schedule: Cosine annealing, 200 epochs
```

---

## Installation

```bash
# Clone this repository
git clone https://github.com/AMAAI-Lab/MERIT.git
cd MERIT

# Create conda environment
conda create -n merit python=3.10 -y
conda activate merit

# Install dependencies
pip install -r requirements.txt
```

### JASCO (required for triplet generation only)

Triplet generation uses the [JASCO](https://github.com/facebookresearch/audiocraft/blob/main/docs/JASCO.md) music generation model (Meta AI). Follow their installation instructions and then set:

```bash
export JASCO_ROOT=/path/to/jasco-audiocraft
```

> **Note:** JASCO is only needed to *re-generate* triplets.

---

## Data Setup

### MoisesDB

1. Request access and download [MoisesDB v0.1](https://music.ai/research/moisesdb/) from Moises Inc.
2. Unpack so that the structure is:
   ```
   /your/path/moisesdb/moisesdb_v0.1/<song_id>/...
   ```
3. Export the environment variable:
   ```bash
   export MOISESDB_ROOT=/your/path/moisesdb
   ```

### Probe Datasets (for Step 6)

Download the three probe datasets and place them under a common root:

| Dataset | Used for | Source |
|---|---|---|
| [MUSDB18-HQ](https://zenodo.org/record/3338373) | Timbral probe | Zenodo |
| [Ballroom](http://mtg.upf.edu/ismir2004/contest/tempoContest/node5.html) | Rhythmic probe | MTG-UPF |
| [Covers80](https://labrosa.ee.columbia.edu/projects/coversongs/covers80/) | Melodic/cover probe | LabROSA |

```bash
export PROBES_ROOT=/your/path/probes
# Expected layout:
#   $PROBES_ROOT/musdb18hq/   (stems as .wav)
#   $PROBES_ROOT/ballroom/    (subdirs named by class)
#   $PROBES_ROOT/covers80/    (subdirs with 2 files = one cover pair)
```

---

## Reproduction

### Using Pre-trained Heads (recommended)

The three trained projection heads (melody, rhythm, timbre) are available on HuggingFace (~11 MB each):

```bash
huggingface-cli download amaai-lab/merit head_mel/best_head.pt head_rhy/best_head.pt head_tim/best_head.pt --local-dir ./models
```

> **Want to run MERIT on your own audio?** This is all you need — no training required. Download the heads, encode your audio with `evaluation/encode_folder.py`, and project with the heads. No MoisesDB, no JASCO, no GPU-days of training.

To reproduce the paper evaluations:

```bash
# 3×3 disentanglement table (Table 1)
export EMBEDDINGS_DIR=./data/embeddings
bash scripts/3_extract_embeddings.sh
bash scripts/5_evaluate.sh

# Probe evaluations (Table 2 / Table 3)
export PROBES_ROOT=/your/path/probes
bash scripts/6_run_probes.sh
```

### Full Reproduction (re-generate everything from scratch)

```bash
# Step 1: Build MoisesDB input indexes
export MOISESDB_ROOT=/your/path/moisesdb
bash scripts/1_build_indexes.sh

# Step 2: Generate triplets (requires JASCO)
export JASCO_ROOT=/path/to/jasco-audiocraft
bash scripts/2_generate_triplets.sh

# Step 3: Extract MERT embeddings
bash scripts/3_extract_embeddings.sh

# Step 4: Train heads
bash scripts/4_train_heads.sh

# Step 5: Evaluate (3×3 disentanglement table)
bash scripts/5_evaluate.sh

# Step 6: Probe evaluations
export PROBES_ROOT=/your/path/probes
bash scripts/6_run_probes.sh
```

### Multi-GPU Embedding Extraction (Optional — Advanced)

`extract_embeddings.py` supports sharding across multiple GPUs to speed up extraction. Skip this if running on a single GPU — `scripts/3_extract_embeddings.sh` handles that directly.

```bash
# Run on 4 GPUs (adjust CUDA_VISIBLE_DEVICES accordingly)
for I in 1 2 3 4; do
  CUDA_VISIBLE_DEVICES=$((I-1)) python training/extract_embeddings.py \
    --encoder mert --triplets-dir ./data/melody_triplets \
    --split-file splits/melody_split.json \
    --out ./data/embeddings/mel_shard_${I}.pkl \
    --shard ${I}/4 &
done
wait

# Merge shards
python training/merge_pkl.py \
  --shards ./data/embeddings/mel_shard_*.pkl \
  --triplets-dir ./data/melody_triplets \
  --out ./data/embeddings/mel_mert.pkl
```

---

## Citation

If you use this code, please cite:

```bibtex
TODO: add after arxiv submission
```

---

## License

This code is released under the [MIT License](LICENSE).

The datasets used (MoisesDB, MUSDB18-HQ, Ballroom, Covers80) are subject to their own respective licenses. See each dataset's homepage for terms of use.
