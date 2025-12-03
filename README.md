# Sparse CLT: Cross-Layer Transcoder Feature Extraction

[![PyPI version](https://badge.fury.io/py/sparse-clt.svg)](https://badge.fury.io/py/sparse-clt)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

An optimized PyTorch library for extracting sparse interpretable features from transformer models using **Cross-Layer Transcoders (CLTs)**.

**Author:** Fahad Alghanim  
**Installation:** `pip install sparse-clt`

---

## What are Cross-Layer Transcoders?

Cross-Layer Transcoders (CLTs) are sparse autoencoders trained to predict **multiple future layers** from a single layer's residual stream:

```
Layer L:  h_L → Encoder → sparse features f → Decoders → [ŷ_{L+1}, ŷ_{L+2}, ..., ŷ_{L+n}]
```

Unlike standard SAEs that reconstruct the same layer, CLTs learn features that are **causally relevant** to the model's computation across multiple layers.

### This Library

This is an **inference library** that uses only the encoder portion of trained CLTs:

```
Hidden State h ∈ ℝ^d  →  f = ReLU(W_enc @ LayerNorm(h) + b_enc)  →  Sparse Features f ∈ ℝ^m
```

The decoder (used during training) is not needed for feature extraction – we only care about which interpretable features activate, not reconstructing MLP outputs.

---

## Key Features

- **Batched CLT Encoding** – Process multiple layers simultaneously
- **Memory Efficient** – Automatic chunking for sequences up to 2048+ tokens
- **Top-K Extraction** – Get only the most activated features per position
- **GPU Optimized** – Vectorized operations, no Python loops
- **Simple API** – Works with any transformer model (LLM/VLM)
- **Configurable** – Threshold filtering, top-k control, batch sizes

---

## Installation

```bash
pip install sparse-clt
```

**Requirements:**
- Python 3.8+
- PyTorch 2.0+
- CUDA-capable GPU (optional but recommended)

---

## Quick Start

```python
import torch
from sparse_clt import SparseCLTEncoder, load_transcoders

# Load trained CLT weights (from vlm-clt-training or similar)
transcoders = load_transcoders(
    transcoder_dir='./clt_checkpoints',
    layers=[0, 1, 2, 3, 4],
    device='cuda'
)

# Create encoder
encoder = SparseCLTEncoder(
    transcoders=transcoders,
    top_k=50,                     # Top 50 features per position
    activation_threshold=1.0,     # Minimum activation value
    chunk_size=512                # Process 512 tokens at a time
)

# Extract features from hidden states
hidden_states = {
    0: torch.randn(1, 256, 4096, device='cuda'),  # [batch, seq, hidden]
    1: torch.randn(1, 256, 4096, device='cuda'),
    # ... more layers
}

# Get sparse features (batched across all layers)
features = encoder.encode_all_layers(hidden_states)

# Access results
for layer_idx, layer_features in features.items():
    print(f"Layer {layer_idx}:")
    print(f"  Activations shape: {layer_features['activations'].shape}")  # [B, T, top_k]
    print(f"  Indices shape: {layer_features['indices'].shape}")          # [B, T, top_k]
    print(f"  Sparsity: {layer_features['sparsity'].mean():.3f}")
```

---

## How It Works

### CLT Architecture (Training)

During training, CLTs learn to predict multiple future MLP outputs:

```
Input:    Residual stream at layer L
Encoder:  LayerNorm → Linear → ReLU/TopK → Sparse features
Decoder:  Linear projections to layers L+1, L+2, ..., L+n
Loss:     Σ MSE(decoder[i](features), MLP_output[L+i])
```

### This Library (Inference)

For feature extraction, we only need the encoder:

```
Hidden States [B, T, H]
       ↓
Encoder: W_enc @ LayerNorm(h) + b_enc
       ↓
Activation: ReLU (natural sparsity)
       ↓
Top-K Selection: Keep K highest activations
       ↓
Sparse Features [B, T, K]
```

---

## Advanced Usage

### Memory-Efficient Long Sequences

```python
encoder = SparseCLTEncoder(
    transcoders=transcoders,
    chunk_size=512  # Automatically chunks long sequences
)

# Works with any sequence length
long_hidden = torch.randn(1, 2048, 4096, device='cuda')
features = encoder.encode_layer(0, long_hidden)
```

### Custom Thresholding

```python
encoder = SparseCLTEncoder(
    transcoders=transcoders,
    activation_threshold=2.0,  # Higher threshold = sparser output
    top_k=100
)
```

### Attribution Graph Features

```python
graph_features = encoder.extract_attribution_features(
    hidden_states=hidden_states,
    top_k_global=100  # Top 100 features across all positions
)

for feature in graph_features[:5]:
    print(f"Layer {feature['layer']}, Feature {feature['feature']}: {feature['activation']:.2f}")
```

---

## API Reference

### `SparseCLTEncoder`

```python
encoder = SparseCLTEncoder(
    transcoders: Dict[int, TranscoderWeights],
    top_k: int = 20,
    activation_threshold: float = 1.0,
    use_compile: bool = True,
    chunk_size: int = 512
)
```

**Methods:**

| Method | Description |
|--------|-------------|
| `encode_layer(layer_idx, hidden)` | Encode single layer |
| `encode_all_layers(hidden_states)` | Encode multiple layers (batched) |
| `encode_chunked(layer_idx, hidden)` | Memory-efficient for long sequences |
| `extract_attribution_features(...)` | Format for attribution graphs |

**Returns:**
```python
{
    'layer': int,
    'activations': torch.Tensor,  # [B, T, top_k]
    'indices': torch.Tensor,      # [B, T, top_k]
    'sparsity': torch.Tensor      # [B, T]
}
```

---

## Use Cases

### Model Interpretability

```python
features = encoder.encode_layer(layer_idx=25, hidden=hidden_states)
top_features = features['indices'][0, -1, :10]  # Top 10 at last position
print(f"Most active features: {top_features}")
```

### Feature Steering

```python
# Find strongly activated features for intervention
features = encoder.encode_all_layers(hidden_states)
for layer, data in features.items():
    strong = data['indices'][data['activations'] > 5.0]
    print(f"Layer {layer}: {len(strong)} strong features")
```

---

## Related Projects

- [**vlm-clt-training**](https://github.com/KOKOSde/vlm-clt-training) – Train CLTs for LLMs/VLMs
- [**EleutherAI/clt-training**](https://github.com/EleutherAI/clt-training) – Original CLT training code
- [**Anthropic Circuit-Tracer**](https://transformer-circuits.pub/2025/attribution-graphs/methods.html) – Attribution graph methodology

---

## Citation

```bibtex
@software{alghanim2025sparseclt,
  author = {Alghanim, Fahad},
  title = {Sparse CLT: Cross-Layer Transcoder Feature Extraction},
  year = {2025},
  url = {https://github.com/KOKOSde/sparse-clt}
}
```

---

## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/improvement`)
3. Add tests for new functionality
4. Submit a pull request

---

## License

MIT License – see [LICENSE](LICENSE) for details.

---

## Contact

**Fahad Alghanim**  
GitHub: [@KOKOSde](https://github.com/KOKOSde)
