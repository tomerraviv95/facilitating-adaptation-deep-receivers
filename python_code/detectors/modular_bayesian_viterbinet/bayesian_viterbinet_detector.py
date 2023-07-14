from typing import Tuple

import torch
import torch.nn as nn

from python_code import DEVICE
from python_code.detectors.viterbinet.viterbinet_detector import create_transition_table, acs_block
from python_code.utils.bayesian_utils import LossVariable
from python_code.utils.constants import Phase

HIDDEN_SIZE = 75


def entropy(prob):
    return -prob * torch.log2(prob) - (1 - prob) * torch.log2(1 - prob)


def dropout_ori(x, logit, u):
    dropout_prob = torch.sigmoid(logit)
    z = (u < dropout_prob).float()
    return x * z


def dropout_tilde(x, logit, u):
    dropout_prob_tilde = torch.sigmoid(-logit)
    z_tilde = (u > dropout_prob_tilde).float()
    return x * z_tilde


class BayesianDNN(nn.Module):
    """
    This Bayesian Implementation follows the paper
    "Learnable Bernoulli Dropout for Bayesian Deep Learning"
    by computing the forward pass by averaging multiple viterbinets runs, each with a different dropout mask

    To learn the dropout parameters we compute additional outputs such as KL term and output_tilde
    which will be used in the computed loss
    """

    def __init__(self, n_states: int, kl_scale: float, ensemble_num: int):
        super(BayesianDNN, self).__init__()
        self.fc1 = nn.Linear(1, HIDDEN_SIZE).to(DEVICE)
        self.activation = nn.ReLU().to(DEVICE)
        self.fc2 = nn.Linear(HIDDEN_SIZE, n_states).to(DEVICE)
        self.dropout_logit = nn.Parameter(torch.rand(HIDDEN_SIZE).reshape(1, -1))
        self.sigmoid = nn.Sigmoid()
        self.log_softmax = nn.LogSoftmax(dim=1)
        self.kl_scale = kl_scale
        self.ensemble_num = ensemble_num

    def forward(self, raw_input: torch.Tensor, phase: Phase):
        log_probs = 0
        arm_original, arm_tilde, u_list, kl_term = [], [], [], 0

        for ind_ensemble in range(self.ensemble_num):
            # first layer
            x = self.activation(self.fc1(raw_input))
            u = torch.rand(x.shape).to(DEVICE)
            x_after_dropout = dropout_ori(x, self.dropout_logit, u)
            # second layer
            out = self.fc2(x_after_dropout)
            # if in train phase, keep parameters in list and compute the tilde output for arm loss calculation
            if phase == Phase.TRAIN:
                log_probs += self.log_softmax(out)
                u_list.append(u)
                # compute first variable output
                arm_original.append(self.log_softmax(out))
                # compute second variable output
                x_tilde = dropout_tilde(x, self.dropout_logit, u)
                out_tilde = self.fc2(x_tilde)
                arm_tilde.append(self.log_softmax(out_tilde))
            else:
                log_probs += self.log_softmax(out)

        log_probs /= self.ensemble_num

        # add KL term if training
        if phase == Phase.TRAIN:
            # KL term
            scaling1 = (self.kl_scale ** 2 / 2) * (torch.sigmoid(self.dropout_logit).reshape(-1))
            first_layer_kl = scaling1 * torch.norm(self.fc1.weight, dim=1) ** 2
            H1 = entropy(torch.sigmoid(self.dropout_logit).reshape(-1))
            kl_term = torch.mean(first_layer_kl - H1)
            return LossVariable(priors=log_probs, arm_original=arm_original, arm_tilde=arm_tilde,
                                u_list=u_list, kl_term=kl_term, dropout_logit=self.dropout_logit)
        return log_probs


class BayesianVNETDetector(nn.Module):
    """
    This implements the Bayesian version of VA decoder by a parameterization of the cost calculation by an NN for each stage
    """

    def __init__(self, n_states: int, kl_scale: float, ensemble_num: int):
        super(BayesianVNETDetector, self).__init__()
        self.n_states = n_states
        self.transition_table_array = create_transition_table(n_states)
        self.transition_table = torch.Tensor(self.transition_table_array).to(DEVICE)
        self.net = BayesianDNN(self.n_states, kl_scale, ensemble_num).to(DEVICE)

    def forward(self, rx: torch.Tensor, phase: Phase) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:

        # initialize input probabilities
        in_prob = torch.zeros([1, self.n_states]).to(DEVICE)

        if phase == Phase.TEST:
            priors = self.net(rx, phase)
            detected_word = torch.zeros(rx.shape).to(DEVICE)
            for i in range(rx.shape[0]):
                # get the lsb of the state
                detected_word[i] = torch.argmin(in_prob, dim=1) % 2
                # run one Viterbi stage
                out_prob = acs_block(in_prob, -priors[i], self.transition_table, self.n_states)
                # update in-probabilities for next layer
                in_prob = out_prob

            return detected_word
        else:
            return self.net(rx, phase)
