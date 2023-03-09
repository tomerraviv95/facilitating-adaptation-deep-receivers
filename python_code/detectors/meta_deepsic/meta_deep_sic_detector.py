from typing import List

import torch
from torch import nn
from torch.nn import functional as F

from python_code.utils.config_singleton import Config

conf = Config()

HIDDEN_BASE_SIZE = 64


class MetaDeepSICDetector(nn.Module):
    """
    The Meta DeepSIC Network Architecture

    ===========Architecture=========
    DeepSICNet(
      (fullyConnectedLayer): Linear(in_features=s_nK+s_nN-1, out_features=60, bias=True)
      (sigmoid): Sigmoid()
      (fullyConnectedLayer): Linear(in_features=60, out_features=30, bias=True)
      (reluLayer): ReLU()
      (fullyConnectedLayer2): Linear(in_features=30, out_features=2, bias=True)
    ================================
    Note:
    The output of the network is not probabilities,
    to obtain probabilities apply a softmax function to the output, viz.
    output = DeepSICNet(data)
    probs = torch.softmax(output, dim), for a batch inference, set dim=1; otherwise dim=0.
    """

    def __init__(self):
        super(MetaDeepSICDetector, self).__init__()
        self.activation = nn.ReLU()

    def forward(self, y: torch.Tensor, var: List[torch.Tensor]) -> torch.Tensor:
        fc_out0 = F.linear(y, var[0], var[1])
        out0 = self.activation(fc_out0)
        fc_out1 = F.linear(out0, var[2], var[3])
        return fc_out1
