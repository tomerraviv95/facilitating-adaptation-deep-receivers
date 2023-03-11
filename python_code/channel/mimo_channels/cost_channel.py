import os

import numpy as np
import scipy.io

from dir_definitions import COST2100_DIR
from python_code.utils.config_singleton import Config

conf = Config()

SCALING_COEF = 0.5
MAX_FRAMES = 25


class Cost2100MIMOChannel:
    @staticmethod
    def calculate_channel(n_ant: int, n_user: int, frame_ind: int, fading: bool) -> np.ndarray:
        total_h = np.empty([n_user, n_ant])
        main_folder = 1 + (frame_ind // MAX_FRAMES)
        for i in range(1, n_user + 1):
            path_to_mat = os.path.join(COST2100_DIR, f'{main_folder}', f'h_{i}.mat')
            h_user = scipy.io.loadmat(path_to_mat)['norm_channel'][frame_ind % MAX_FRAMES, :conf.n_user]
            total_h[i - 1] = SCALING_COEF * h_user

        total_h[np.arange(n_user), np.arange(n_user)] = 1
        return total_h

    @staticmethod
    def transmit(s: np.ndarray, h: np.ndarray, snr: float) -> np.ndarray:
        """
        The MIMO COST2100 Channel
        :param s: to transmit symbol words
        :param snr: signal-to-noise value
        :param h: channel coefficients
        :return: received word
        """
        conv = Cost2100MIMOChannel._compute_channel_signal_convolution(h, s)
        sigma = 10 ** (-0.1 * snr)
        w = np.sqrt(sigma) * np.random.randn(conf.n_ant, s.shape[1])
        y = conv + w
        return y

    @staticmethod
    def _compute_channel_signal_convolution(h: np.ndarray, s: np.ndarray) -> np.ndarray:
        conv = np.matmul(h, s)
        return conv
