## Implement the LBD method "Learnable Bernoulli Dropout for Bayesian Deep Learning"

from typing import Union

import torch
from torch import nn

from python_code import DEVICE, conf
from python_code.channel.modulator import MODULATION_NUM_MAPPING
from python_code.utils.bayesian_utils import dropout_ori, dropout_tilde, entropy, LossVariable
from python_code.utils.constants import Phase, ModulationType

HIDDEN_BASE_SIZE = 64


class BayesianDeepSICDetector(nn.Module):

    def __init__(self, ensemble_num, kl_scale):
        super(BayesianDeepSICDetector, self).__init__()
        classes_num = MODULATION_NUM_MAPPING[conf.modulation_type]
        hidden_size = HIDDEN_BASE_SIZE * classes_num
        base_rx_size = conf.n_ant if conf.modulation_type == ModulationType.BPSK.name else 2 * conf.n_ant
        linear_input = base_rx_size + (classes_num - 1) * (conf.n_user - 1)  # from DeepSIC paper
        self.fc1 = nn.Linear(linear_input, hidden_size)
        self.activation = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, classes_num)
        self.ensemble_num = ensemble_num
        self.kl_scale = kl_scale
        self.dropout_logit = nn.Parameter(torch.rand(hidden_size).reshape(1, -1))
        self.log_softmax = nn.LogSoftmax(dim=1)

    def forward(self, rx: torch.Tensor, phase: Phase = Phase.TEST) -> Union[LossVariable, torch.Tensor]:
        log_probs = 0
        arm_original, arm_tilde, u_list, kl_term = [], [], [], 0

        for ind_ensemble in range(self.ensemble_num):
            # first layer
            x = self.activation(self.fc1(rx))
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
