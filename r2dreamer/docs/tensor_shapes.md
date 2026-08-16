# Tensor Shape Hints

To make the code easier to understand, some comments include **inline tensor shape annotations** using the symbols below.

| Symbol | Meaning |
| --- | --- |
| `B` | Batch size (e.g., number of parallel environments or replay batch size) |
| `T` | Time steps (sequence length) |
| `T_imag` | Imagination horizon (time steps in latent rollouts) |
| `A` | Action dimension |
| `E` | Encoder embedding dimension |
| `F` | RSSM feature dimension (`S*K + D`) |
| `S` | Number of stochastic groups |
| `K` | Number of discrete categories per stochastic group |
| `D` | Deterministic state dimension |
| `G` | Number of groups/blocks (e.g., BlockLinear groups) |
| `U` | Hidden units (generic intermediate dimension) |
| `H`, `W` | Image height and width |
| `C` | Image channels |
| `H_feat`, `W_feat`, `C_feat` | Intermediate feature map shape (spatial resolution and channels) inside CNN modules |
