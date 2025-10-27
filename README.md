# 🚀 Sparse CLT: Cross-Layer Transcoder Feature Extraction

[![PyPI version](https://badge.fury.io/py/sparse-clt.svg)](https://badge.fury.io/py/sparse-clt)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

**Author:** Fahad Alghanim  
**Performance:** Batched sparse feature extraction using Cross-Layer Transcoders  
**Installation:** `pip install sparse-clt`

---

## 🎯 What is This?

**Sparse CLT** is an optimized PyTorch library for extracting sparse interpretable features from transformer models using **Cross-Layer Transcoders (CLTs)**.

### What are Cross-Layer Transcoders?

Cross-Layer Transcoders (CLTs) are neural networks trained to decompose dense MLP activations into sparse, interpretable features:

```
h ∈ ℝ^d → f = ReLU(W_enc @ h + b_enc) ∈ ℝ^m (sparse features)
```

Unlike standard autoencoders that reconstruct the same layer, CLTs predict the **next layer's activations**, learning features that are causally relevant to the model's computation.

### This Library Provides:

- **Model Interpretability** - Extract interpretable features from any LLM/VLM
- **Attribution Analysis** - Find which CLT features influence specific outputs  
- **Efficient Processing** - Batched operations across 20+ layers simultaneously
- **Production Ready** - Fast inference, memory-efficient, well-tested

---

## ⚡ Key Features

- 🔥 **Batched CLT Encoding** - Process multiple layers simultaneously
- 💾 **Memory Efficient** - Automatic chunking for sequences up to 2048+ tokens
- 🎯 **Top-K Extraction** - Get only the most activated features per position
- 🚀 **GPU Optimized** - Vectorized operations, no Python loops
- 📦 **Easy to Use** - Simple API, works with any transformer model
- 🔧 **Configurable** - Threshold filtering, top-k control, batch sizes

---

## 📦 Installation

```bash
pip install sparse-clt
```

**Requirements:**
- Python 3.8+
- PyTorch 2.0+
- CUDA-capable GPU (optional but recommended)

---

## 🚀 Quick Start

```python
import torch
from sparse_clt import SparseCLTEncoder, load_transcoders

# Load your trained CLT weights
transcoders = load_transcoders(
    transcoder_dir='/path/to/clt_checkpoints',
    layers=[40, 41, 42, 43, 44],  # Which layers have CLTs
    device='cuda'
)

# Create encoder
encoder = SparseCLTEncoder(
    transcoders=transcoders,
    top_k=50,                     # Top 50 features per position
    activation_threshold=1.0,      # Minimum activation value
    chunk_size=512                 # Process 512 tokens at a time
)

# Extract CLT features from hidden states
hidden_states = {
    40: torch.randn(1, 256, 5120, device='cuda'),  # [batch, seq, hidden]
    41: torch.randn(1, 256, 5120, device='cuda'),
    # ... more layers
}

# Get sparse features (batched across all layers!)
features = encoder.encode_all_layers(hidden_states)

# Access results
for layer_idx, layer_features in features.items():
    print(f"Layer {layer_idx}:")
    print(f"  Activations: {layer_features['activations'].shape}")  # [B, T, top_k]
    print(f"  Indices: {layer_features['indices'].shape}")          # [B, T, top_k]
    print(f"  Sparsity: {layer_features['sparsity'].mean():.3f}")   # Fraction active
```

---

## 📊 Performance

### Why Use This Library?

| Feature | Manual Loops | Sparse CLT |
|---------|-------------|------------|
| API simplicity | ❌ Manual loops | ✅ One function call |
| Memory efficiency | ❌ OOM on long seq | ✅ Handles any length |
| Batched processing | ❌ Sequential | ✅ All layers at once |
| Production ready | ❌ Research code | ✅ Tested & documented |

**Speed:** Equivalent to manual implementation (~45ms per layer) but with much cleaner API and automatic memory management.

---

## 🔬 Advanced Usage

### Memory-Efficient Long Sequences

```python
# Automatically chunks long sequences
encoder = SparseCLTEncoder(
    transcoders=transcoders,
    chunk_size=512  # Process 512 tokens at a time
)

# Works with sequences of any length!
long_hidden = torch.randn(1, 2048, 5120, device='cuda')  # 2048 tokens
features = encoder.encode_layer(40, long_hidden)
```

### Custom Thresholding

```python
# Only keep features above threshold
encoder = SparseCLTEncoder(
    transcoders=transcoders,
    activation_threshold=2.0,  # Higher threshold = sparser
    top_k=100
)
```

### Extract for Attribution Graphs

```python
# Get features formatted for attribution analysis
graph_features = encoder.extract_attribution_features(
    hidden_states=hidden_states,
    top_k_global=100  # Top 100 features across all positions
)

# Returns list of dicts ready for graph generation
for feature in graph_features[:5]:
    print(f"Layer {feature['layer']}, Feature {feature['feature']}: {feature['activation']:.2f}")
```

---

## 🏗️ How Cross-Layer Transcoders Work

### Standard SAE (Same-Layer Reconstruction)
```
Layer N:  h_n → SAE → reconstruct h_n
Problem:  Features may not be causally relevant
```

### Cross-Layer Transcoder (CLT)
```
Layer N:    h_n → CLT → predict h_{n+1}
Layer N+1:  actual h_{n+1}
Loss:       ||CLT(h_n) - h_{n+1}||²

Advantage:  Features must be causally relevant to predict next layer
```

### This Library's Role

```
Input: Hidden States from Model [B, T, H]
         ↓
CLT Encode: h @ W_enc.T + b_enc
         ↓
Activation: ReLU(features)
         ↓
Sparse CLT: Threshold + Top-K
         ↓
Output: Interpretable Features [B, T, K]
```

---

## 📚 API Reference

### `SparseCLTEncoder`

**Constructor:**
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

- `encode_layer(layer_idx, hidden)` - Encode single layer with CLT
- `encode_all_layers(hidden_states)` - Encode multiple layers (batched)
- `encode_chunked(layer_idx, hidden)` - Memory-efficient for long sequences
- `extract_attribution_features(hidden_states, top_k_global)` - For graph generation

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

## 🎓 Use Cases

### 1. Model Interpretability

Understand which CLT features activate in your model:

```python
features = encoder.encode_layer(layer_idx=25, hidden=hidden_states)
top_features = features['indices'][0, -1, :10]  # Top 10 at last position
print(f"Most active CLT features: {top_features}")
```

### 2. Attribution Analysis

Find CLT features that influence specific outputs:

```python
graph_features = encoder.extract_attribution_features(
    hidden_states=all_layers,
    top_k_global=100
)
# Use in attribution graph generation
```

### 3. Feature Steering

Identify CLT features to amplify/suppress for behavior modification:

```python
# Find strongly activated features
features = encoder.encode_all_layers(hidden_states)
for layer, data in features.items():
    strong_features = data['indices'][data['activations'] > 5.0]
    print(f"Layer {layer}: {len(strong_features)} strong CLT features")
```

---

## 🔧 Development

### Running Tests

```bash
git clone https://github.com/KOKOSde/sparse-clt.git
cd sparse-clt
pip install -e ".[dev]"
pytest tests/
```

### Building from Source

```bash
pip install build
python -m build
```

---

## 📖 Citation

If you use Sparse CLT in your research, please cite:

```bibtex
@software{alghanim2025sparseclt,
  author = {Alghanim, Fahad},
  title = {Sparse CLT: Cross-Layer Transcoder Feature Extraction for Transformer Models},
  year = {2025},
  url = {https://github.com/KOKOSde/sparse-clt}
}
```

---

## 🤝 Contributing

Contributions welcome! Please:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/improvement`)
3. Add tests for new functionality
4. Ensure all tests pass (`pytest tests/`)
5. Submit a pull request

---

## 📄 License

MIT License - see LICENSE file for details.

---

## 📧 Contact

**Fahad Alghanim**  
Email: fkalghan@email.sc.edu  
GitHub: [@KOKOSde](https://github.com/KOKOSde)

---

## 🌟 Related Work

- **Cross-Layer Transcoders** - Original method for causally-relevant feature extraction
- **Mechanistic Interpretability** - Understanding how models work internally
- **Sparse Autoencoders (SAEs)** - Same-layer reconstruction (this library uses CLTs instead)

---

<div align="center">

**⚡ Fast | 🎯 Accurate | 📦 Easy to Use**

[Documentation](https://github.com/KOKOSde/sparse-clt) | [PyPI](https://pypi.org/project/sparse-clt/) | [Issues](https://github.com/KOKOSde/sparse-clt/issues)

</div>
