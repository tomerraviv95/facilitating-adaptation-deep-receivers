## Bayesian utils functions to implement the LBD method "Learnable Bernoulli Dropout for Bayesian Deep Learning"

import collections

import torch

LossVariable = collections.namedtuple('LossVariable', 'priors u_list arm_original arm_tilde dropout_logit kl_term')


def entropy(prob: torch.Tensor) -> torch.Tensor:
    return -prob * torch.log2(prob) - (1 - prob) * torch.log2(1 - prob)


def dropout_ori(x: torch.Tensor, logit: torch.Tensor, u: torch.Tensor) -> torch.Tensor:
    dropout_prob = torch.sigmoid(logit)
    z = (u < dropout_prob).float()
    return x * z


def dropout_tilde(x: torch.Tensor, logit: torch.Tensor, u: torch.Tensor) -> torch.Tensor:
    dropout_prob_tilde = torch.sigmoid(-logit)
    z_tilde = (u > dropout_prob_tilde).float()
    return x * z_tilde
