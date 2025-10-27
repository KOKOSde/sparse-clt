"""
Basic usage example for sparse-transcoder

Shows how to extract sparse features from transformer activations
"""

import torch
from sparse_transcoder import SparseCLTEncoder, TranscoderWeights


def example_basic():
    """Basic feature extraction"""
    print("="*70)
    print("Example 1: Basic Feature Extraction")
    print("="*70)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}\n")
    
    # Create example transcoders (in real use, load from checkpoint)
    transcoders = {
        40: TranscoderWeights(
            W_enc=torch.randn(12288, 5120, device=device),
            b_enc=torch.zeros(12288, device=device),
            W_dec=torch.randn(5120, 12288, device=device),
            b_dec=torch.zeros(5120, device=device),
            layer=40
        ),
        41: TranscoderWeights(
            W_enc=torch.randn(12288, 5120, device=device),
            b_enc=torch.zeros(12288, device=device),
            W_dec=torch.randn(5120, 12288, device=device),
            b_dec=torch.zeros(5120, device=device),
            layer=41
        )
    }
    
    # Create encoder
    encoder = SparseCLTEncoder(
        transcoders=transcoders,
        top_k=50,
        activation_threshold=1.0
    )
    
    # Example hidden states
    hidden_states = {
        40: torch.randn(1, 256, 5120, device=device),
        41: torch.randn(1, 256, 5120, device=device),
    }
    
    # Extract features
    features = encoder.encode_all_layers(hidden_states)
    
    # Print results
    for layer_idx, layer_features in features.items():
        print(f"Layer {layer_idx}:")
        print(f"  Activations shape: {layer_features['activations'].shape}")
        print(f"  Top feature indices: {layer_features['indices'][0, 0, :5]}")
        print(f"  Sparsity: {layer_features['sparsity'].mean():.3f}")
        print()


def example_attribution():
    """Extract features for attribution graphs"""
    print("="*70)
    print("Example 2: Attribution Graph Features")
    print("="*70)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Create transcoders
    transcoders = {
        i: TranscoderWeights(
            W_enc=torch.randn(8192, 3584, device=device),
            b_enc=torch.zeros(8192, device=device),
            W_dec=torch.randn(3584, 8192, device=device),
            b_dec=torch.zeros(3584, device=device),
            layer=i
        )
        for i in range(20, 25)
    }
    
    encoder = SparseCLTEncoder(transcoders, top_k=30)
    
    # Hidden states
    hidden_states = {
        i: torch.randn(1, 128, 3584, device=device)
        for i in range(20, 25)
    }
    
    # Extract for graph
    graph_features = encoder.extract_attribution_features(
        hidden_states,
        top_k_global=50
    )
    
    print(f"Extracted {len(graph_features)} graph features\n")
    print("Top 5 features:")
    for feature in graph_features[:5]:
        print(f"  Layer {feature['layer']}, Feature {feature['feature']}: "
              f"{feature['activation']:.2f} at position {feature['ctx_idx']}")


if __name__ == '__main__':
    example_basic()
    print("\n")
    example_attribution()

