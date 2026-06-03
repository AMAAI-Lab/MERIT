<div align="center">

<h1>MERIT</h1>
<h3>Multi-Factor Disentangled Music Similarity</h3>

[![arXiv](https://img.shields.io/badge/arXiv-2605.27346-b31b1b.svg)](https://arxiv.org/abs/2605.27346)
[![HuggingFace Models](https://img.shields.io/badge/%F0%9F%A4%97%20Models-amaai--lab%2Fmerit-ffd21e?style=flat-square)](https://huggingface.co/amaai-lab/merit)
[![HuggingFace Dataset](https://img.shields.io/badge/%F0%9F%A4%97%20Dataset-amaai--lab%2Fmerit-ff9d00?style=flat-square)](https://huggingface.co/datasets/amaai-lab/merit)

<br/>

*Most similarity models collapse melody, rhythm, and timbre into a single undifferentiated score.*
*MERIT exposes all three as independent, interpretable signals from the same audio query.* 📄 **[Paper TLDR](https://arxivexplained.com/papers/merit-learning-disentangled-music-representations-for-audio-similarity)**

<br/>

![MERIT architecture](./fig_overview.png)

</div>

---

## What is MERIT?

Given two audio clips, MERIT returns **three independent cosine similarities** — one per musical factor:

| Score | Captures | Example query |
|---|---|---|
| `S_mel` | Melodic contour & pitch identity | *"Find songs with the same melody"* |
| `S_rhy` | Rhythmic groove & beat pattern | *"Find songs with the same drum feel"* |
| `S_tim` | Instrument timbre & sonic character | *"Find songs played on the same instrument"* |

A solo piano cover of a rock anthem scores high on `S_mel` but low on `S_rhy` and `S_tim`. MERIT makes this distinction explicit and computable.

---

## HuggingFace Resources

| Resource | Link | Description |
|---|---|---|
| Pre-trained heads | [amaai-lab/merit](https://huggingface.co/amaai-lab/merit) | 3 × ~11 MB projection heads |
| Training dataset | [datasets/amaai-lab/merit](https://huggingface.co/datasets/amaai-lab/merit) | 296K factor-controlled triplets |

```bash
# Download pre-trained heads only (~33 MB total)
huggingface-cli download amaai-lab/merit \
    head_mel/best_head.pt head_rhy/best_head.pt head_tim/best_head.pt \
    --local-dir ./models
```

---

## Quick Inference — Get MERIT Embeddings for Your Audio

No training or dataset required. Download the three pre-trained heads (~11 MB each) and encode any audio in a few lines of Python.

### Step 1 — Download pre-trained heads

```bash
pip install torch torchaudio transformers huggingface_hub

huggingface-cli download amaai-lab/merit \
    head_mel/best_head.pt head_rhy/best_head.pt head_tim/best_head.pt \
    --local-dir ./models
```

### Step 2 — Encode audio

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio
from transformers import AutoModel, Wav2Vec2FeatureExtractor

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
EXTRACT_LAYERS = (3, 4, 5, 6, 23)
MODEL_ID = "m-a-p/MERT-v1-330M"

# Load MERT backbone (shared for all three factors)
processor = Wav2Vec2FeatureExtractor.from_pretrained(MODEL_ID, trust_remote_code=True)
mert = AutoModel.from_pretrained(MODEL_ID, trust_remote_code=True).to(DEVICE).eval()


class ProjectionHead(nn.Module):
    def __init__(self, in_dim=5120, hidden_dim=512, out_dim=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, out_dim, bias=False),
        )

    def forward(self, x):
        return F.normalize(self.net(x), dim=-1)


def load_head(path):
    ckpt = torch.load(path, map_location=DEVICE, weights_only=True)
    head = ProjectionHead(ckpt["in_dim"], ckpt["hidden_dim"], ckpt["out_dim"])
    head.load_state_dict(ckpt["state_dict"])
    return head.to(DEVICE).eval()


head_mel = load_head("models/head_mel/best_head.pt")
head_rhy = load_head("models/head_rhy/best_head.pt")
head_tim = load_head("models/head_tim/best_head.pt")


def load_audio(path, sr=24_000, max_sec=30):
    wav, orig_sr = torchaudio.load(path)
    if orig_sr != sr:
        wav = torchaudio.functional.resample(wav, orig_sr, sr)
    wav = wav.mean(0)                                    # stereo → mono
    wav = wav[: sr * max_sec]                            # truncate
    wav = F.pad(wav, (0, sr * max_sec - wav.shape[0]))   # zero-pad
    return wav


@torch.no_grad()
def get_merit_embeddings(audio_path):
    """Return (melody, rhythm, timbre) embeddings — each a (1, 128) unit vector."""
    wav = load_audio(audio_path)
    inputs = processor(wav.numpy(), sampling_rate=24_000, return_tensors="pt")
    inputs = {k: v.to(DEVICE) for k, v in inputs.items()}
    out = mert(**inputs, output_hidden_states=True)
    parts = [out.hidden_states[l].mean(dim=1) for l in EXTRACT_LAYERS]
    backbone = torch.cat(parts, dim=-1)  # (1, 5120)
    return head_mel(backbone), head_rhy(backbone), head_tim(backbone)


# Get embeddings for any two audio files
emb_a = get_merit_embeddings("song_a.wav")
emb_b = get_merit_embeddings("song_b.wav")

melody_sim = (emb_a[0] * emb_b[0]).sum().item()  # cosine sim in [-1, 1]
rhythm_sim  = (emb_a[1] * emb_b[1]).sum().item()
timbre_sim  = (emb_a[2] * emb_b[2]).sum().item()
```

> **Tip:** For large collections, use `evaluation/encode_folder.py` to batch-encode an entire directory to a single pkl file — much faster than encoding file-by-file.

---

## Architecture

```
Audio (24 kHz mono)
  └─► MERT-v1-330M [FROZEN]
        Layers 3, 4, 5, 6, 23
        └─► mean-pool over time → concat → 5120-dim

5120-dim backbone vector
  ├─► H_mel  Linear(5120→512) → ReLU → Linear(512→128) → L2-norm  →  S_mel
  ├─► H_rhy  Linear(5120→512) → ReLU → Linear(512→128) → L2-norm  →  S_rhy
  └─► H_tim  Linear(5120→512) → ReLU → Linear(512→128) → L2-norm  →  S_tim
```

Each head is trained **independently** with Circle Loss on triplets where only one musical factor varies at a time.

| Component | Detail |
|---|---|
| Backbone | MERT-v1-330M, 330M params, frozen |
| Layers extracted | 3, 4, 5, 6, 23 (5 × 1024-dim → 5120-dim) |
| Head architecture | Linear → ReLU → Linear → L2-norm |
| Embedding dim | 128 per factor |
| Loss | Circle Loss (γ=10, m=0.2) |
| Optimizer | AdamW, lr=1e-3 |
| Schedule | Cosine annealing, 200 epochs |

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

## Training Data

The factor-controlled training triplets are published on HuggingFace:

| Factor | Folders | Triplets |
|---|---|---|
| Melody | 5,000 | 125,000 |
| Rhythm | 5,000 | 125,000 |
| Timbre | 1,855 | 46,241 |

```bash
# Download all three factor archives (~50 GB melody, ~50 GB rhythm, ~10 GB timbre)
huggingface-cli download --repo-type dataset amaai-lab/merit \
    melody_triplets.tar.gz rhythm_triplets.tar.gz timbre_triplets.tar.gz \
    --local-dir ./data/triplets
```

Each archive extracts to `triplets_*/triplet/{anchor.wav, positive_01-05.wav, negative.wav, triplet_meta.json}`. Within each folder, **only the target factor is shared** between anchor and positives — key, genre, and instrumentation vary freely.

> Dataset licensed under [CC BY-NC-SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/) — derived from [MoisesDB](https://music.ai/research/moisesdb/). Non-commercial use only.

---

## Data Setup

### MoisesDB

1. Request access and download [MoisesDB](https://github.com/moises-ai/moises-db) from Moises Inc.
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
@article{roy2026merit,
  title={MERIT: Learning Disentangled Music Representations for Audio Similarity},
  author={Roy, Abhinaba and Liang, Junyi and Herremans, Dorien},
  journal={arXiv preprint arXiv:2605.27346},
  year={2026}
}
```

---

## License

This code is released under the [MIT License](LICENSE).

The datasets used (MoisesDB, MUSDB18-HQ, Ballroom, Covers80) are subject to their own respective licenses. See each dataset's homepage for terms of use.
