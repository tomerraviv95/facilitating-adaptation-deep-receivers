import torch
import torch.nn as nn

from python_code import DEVICE, conf
from python_code.utils.constants import ModulationType

INPUT_SIZE = 1
NUM_LAYERS = 2
HIDDEN_SIZE = 64


class RNNDetector(nn.Module):
    """
    This class implements an RNN detector
    """

    def __init__(self, memory_length: int):
        super(RNNDetector, self).__init__()
        self.output_size = 2 ** memory_length
        self.base_rx_size = INPUT_SIZE if conf.modulation_type == ModulationType.BPSK.name else 2 * INPUT_SIZE
        self.lstm = nn.LSTM(INPUT_SIZE, HIDDEN_SIZE, NUM_LAYERS).to(DEVICE)
        self.linear = nn.Linear(HIDDEN_SIZE, self.output_size).to(DEVICE)

    def forward(self, rx: torch.Tensor) -> torch.Tensor:
        """
        The forward pass of the RNN detector
        :param rx: input values, size [batch_size,transmission_length]
        :param phase: 'train' or 'val'
        :return: if in 'train' - the estimated bitwise prob [batch_size,transmission_length,N_CLASSES]
        if in 'val' - the detected words [n_batch,transmission_length]
        """
        # Set initial states
        h_n = torch.zeros(NUM_LAYERS, 1, HIDDEN_SIZE).to(DEVICE)
        c_n = torch.zeros(NUM_LAYERS, 1, HIDDEN_SIZE).to(DEVICE)

        # Forward propagate rnn_out: tensor of shape (seq_length, batch_size, input_size)
        rnn_out, _ = self.lstm(rx.unsqueeze(1), (h_n.contiguous(), c_n.contiguous()))

        # Linear layer output
        out = self.linear(rnn_out.squeeze(1))
        return out
