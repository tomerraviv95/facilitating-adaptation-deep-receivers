import numpy as np
import torch

from python_code.utils.constants import HALF


class BPSKModulator:
    @staticmethod
    def modulate(c: np.ndarray) -> np.ndarray:
        """
        BPSK modulation 0->1, 1->-1
        :param c: the binary codeword
        :return: binary modulated signal
        """
        x = 1 - 2 * c
        return x

    @staticmethod
    def demodulate(s: torch.Tensor) -> torch.Tensor:
        """
        symbol_to_prob(x:PyTorch/Numpy Tensor/Array)
        Converts BPSK Symbols to Probabilities: '-1' -> 0, '+1' -> '1.'
        :param s: symbols vector
        :return: probabilities vector
        """
        return HALF * (s + 1)


class QPSKModulator:
    @staticmethod
    def modulate(c: np.ndarray) -> np.ndarray:
        """
        QPSK modulation
        [0,0] -> [1/sqrt(2),1/sqrt(2)]
        [0,1] -> [1/sqrt(2),-1/sqrt(2)]
        [1,0] -> [-1/sqrt(2),1/sqrt(2)]
        [1,1] -> [-1/sqrt(2),-1/sqrt(2)]
        :param c: the binary codeword
        :return: modulated signal
        """
        x = (-1) ** c[:, ::2] / np.sqrt(2) + (-1) ** c[:, 1::2] / np.sqrt(2) * 1j
        return x

    @staticmethod
    def demodulate(s: torch.Tensor) -> torch.Tensor:
        return ((-1) * HALF * (torch.view_as_real(s) - 1)).transpose(1, 2).reshape(-1, s.shape[1])


MODULATION_DICT = {
    'BPSK': BPSKModulator,
    'QPSK': QPSKModulator,
}

MODULATION_NUM_MAPPING = {
    'BPSK': 2,
    'QPSK': 4,
}
