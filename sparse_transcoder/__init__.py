"""
Sparse Transcoder: Optimized Feature Extraction for Transformer Models
Author: Fahad Alghanim
"""

__version__ = "0.1.0"

from .encoder import (
    SparseTranscoderEncoder,
    TranscoderWeights,
    load_transcoders_from_dir as load_transcoders
)

__all__ = [
    'SparseTranscoderEncoder',
    'TranscoderWeights',
    'load_transcoders',
    '__version__'
]

