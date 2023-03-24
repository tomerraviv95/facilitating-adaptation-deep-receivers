import numpy as np
from numpy.random import default_rng

from python_code import conf

GAMMA = 0.5  # gamma value for time decay SISO fading


class ISIAWGNChannel:
    @staticmethod
    def calculate_channel(memory_length: int, fading: bool = False, index: int = 0) -> np.ndarray:
        h = np.reshape(np.exp(-GAMMA * np.arange(memory_length)), [1, memory_length])
        if fading:
            h = ISIAWGNChannel._add_fading(h, memory_length, index)
        else:
            h *= 0.8
        return h

    @staticmethod
    def _add_fading(h: np.ndarray, memory_length: int, index: int) -> np.ndarray:
        fading_taps = np.array([51, 39, 33, 21])
        h *= (0.8 + 0.2 * np.cos(2 * np.pi * index / fading_taps)).reshape(1, memory_length)
        return h

    @staticmethod
    def transmit(s: np.ndarray, h: np.ndarray, snr: float, memory_length: int) -> np.ndarray:
        """
        The SISO AWGN Channel
        :param s: to transmit symbol words
        :param snr: signal-to-noise value
        :param h: channel function
        :param memory_length: length of channel memory
        :return: received word
        """
        conv = ISIAWGNChannel._compute_channel_signal_convolution(h, memory_length, s)
        [row, col] = conv.shape
        w = ISIAWGNChannel._sample_noise_vector(row, col, snr)
        y = conv + w
        if not conf.linear:
            y = np.tanh(y)
        return y

    @staticmethod
    def _compute_channel_signal_convolution(h: np.ndarray, memory_length: int, s: np.ndarray) -> np.ndarray:
        blockwise_s = np.concatenate([s[:, i:-memory_length + i] for i in range(memory_length)], axis=0)
        conv = np.dot(h[:, ::-1], blockwise_s)
        return conv

    @staticmethod
    def _sample_noise_vector(row: int, col: int, snr: float) -> np.ndarray:
        noise_generator = default_rng(seed=conf.seed)
        snr_value = 10 ** (snr / 10)
        w = (snr_value ** (-0.5)) * noise_generator.standard_normal((row, col))
        return w
