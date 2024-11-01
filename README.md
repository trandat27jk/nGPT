# nGPT
## Paper Summary

**Title**: NGPT: Normalized Transformer with Representation Learning on the Hypersphere  
**Authors**: Ilya Loshchilov, Cheng-Ping Hsieh, Simeng Sun, Boris Ginsburg  
**Abstract**: The NGPT model proposes a novel approach by enforcing unit norm normalization for vectors across the Transformer’s embeddings, MLP, and attention matrices. This hyperspherical projection ensures more stable training, increased efficiency, and faster convergence rates compared to conventional Transformers.

## Key Features

- **Hyperspherical Embeddings**: Projecting vectors onto a hypersphere maintains consistent norms, improving stability.
- **Variable-Metric Optimization**: Layers function as a variable-metric optimizer, with attention and MLP updates guided by eigen learning rates.
- **Reduced Training Cost**: Achieves up to 4x–20x faster convergence, depending on the sequence length, through efficient representation learning.
