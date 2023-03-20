import torch
from torch import nn

from python_code import DEVICE, conf
from python_code.channel.modulator import MODULATION_NUM_MAPPING
from python_code.utils.constants import ModulationType

HIDDEN_SIZE = 60


class DNNDetector(nn.Module):
    """
    The DNNDetector Network Architecture
    """

    def __init__(self, n_user, n_ant):
        super(DNNDetector, self).__init__()
        self.n_user = n_user
        self.n_ant = n_ant
        self.base_rx_size = self.n_ant if conf.modulation_type == ModulationType.BPSK.name else 2 * self.n_ant
        self.n_states = MODULATION_NUM_MAPPING[conf.modulation_type] ** n_ant
        self.initialize_dnn()

    def initialize_dnn(self):
        layers = [nn.Linear(self.base_rx_size, HIDDEN_SIZE),
                  nn.ReLU(),
                  nn.Linear(HIDDEN_SIZE, HIDDEN_SIZE),
                  nn.ReLU(),
                  nn.Linear(HIDDEN_SIZE, HIDDEN_SIZE),
                  nn.ReLU(),
                  nn.Linear(HIDDEN_SIZE, self.n_states)]
        self.net = nn.Sequential(*layers).to(DEVICE)

    def forward(self, rx: torch.Tensor) -> torch.Tensor:
        soft_estimation = self.net(rx)
        return soft_estimation