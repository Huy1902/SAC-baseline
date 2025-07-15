import torch

def dot_scorer(action_emb, item_emb, item_dim):
  '''
  score = item_emb * weight

  @input:
  - action_emb: (B, i_dim)
  - item_emb: (B, L, i_dim) or (1, L, i_dim)
  @output:
  - score: (B, L)
  '''
  output = torch.sum(action_emb.view(-1, 1, item_dim) * item_emb, dim=-1)

  return output