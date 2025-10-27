"""
Sparse Transcoder: Optimized Feature Extraction for Transformer Models
Author: Fahad Alghanim
"""

__version__ = "0.1.0"

from .encoder import (
    SparseCLTEncoder,
    TranscoderWeights,
    load_transcoders as load_transcoders
)

__all__ = [
    'SparseCLTEncoder',
    'TranscoderWeights',
    'load_transcoders',
    '__version__'
]

