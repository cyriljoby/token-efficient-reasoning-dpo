## Base Model Architecture (QWEN2)
**Layer**: each layer applies attention then MLP, with RMSNorm before each and the result added back to the running vector (residual connection)
- Attention: lets each token look at other tokens and pull in relevant information
- MLP: transforms what attention gathered through some computation
- RMSNorm (root-mean-square): Before feeding a vector into attention or the MLP, its rescaled to make training stable

**Attention**: Attention tunes each token's vector into 3 roles:
- Query: What the token is looking for
- Key: What the token contains (used for query matching)
- Value: What the token actually offers once matched 

**MLP**
- After attention, each token goes through a small feedforward network. 
- Typically expansion, nonlinearity, compression
- SwiGLU, which Qwen uses, has two parallel expansions: `up_proj` expands the dimension and carries the actual content, while `gate_proj` (passed through SiLU) is a separate expansion that determines how much of that content is let through
- The MLP dominates parameter counts because MLP briefly epands to a much wider dimensional space

**lm_head**: linear projection from hidden_dim->vocabulary space that expands the last layer's output into one raw score (logit) per vocabulary token. Softmax turns these into next-token probabilities