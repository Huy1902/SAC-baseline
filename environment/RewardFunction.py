import torch

def mean_with_cost(feedback, zero_reward_cost=0.1):
  B, L = feedback.shape
  cost = torch.zeros_like(feedback)
  cost[feedback == 0] = -zero_reward_cost
  reward = torch.mean(feedback + cost, dim=-1)
  return reward