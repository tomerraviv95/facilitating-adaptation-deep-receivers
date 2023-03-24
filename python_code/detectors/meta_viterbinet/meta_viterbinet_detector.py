from typing import Dict

import torch
import torch.nn as nn
from torch.nn import functional as F

from python_code import DEVICE
from python_code.detectors.viterbinet.viterbinet_detector import create_transition_table, acs_block



class MetaViterbiNetDetector(nn.Module):

    def __init__(self, n_states: int):

        super(MetaViterbiNetDetector, self).__init__()
        self.n_states = n_states
        self.transition_table_array = create_transition_table(n_states)
        self.transition_table = torch.Tensor(self.transition_table_array).to(DEVICE)

    def forward(self, rx: torch.Tensor, phase: str, var: list) -> torch.Tensor:
        in_prob = torch.zeros([1, self.n_states]).to(DEVICE)
        # compute priors based on input list of NN parameters
        x = rx.reshape(-1, 1)
        x = F.linear(x, var[0], var[1])
        x = nn.functional.relu(x)
        priors = F.linear(x, var[2], var[3])

        if phase == 'val':
            decoded_word = torch.zeros(rx.shape).to(DEVICE)
            for i in range(rx.shape[0]):
                # get the lsb of the state
                decoded_word[:, i] = torch.argmin(in_prob, dim=1) % 2
                # run one Viterbi stage
                out_prob = acs_block(in_prob, -priors[i], self.transition_table, self.n_states)
                # update in-probabilities for next layer
                in_prob = out_prob

            return decoded_word
        else:
            return priors
