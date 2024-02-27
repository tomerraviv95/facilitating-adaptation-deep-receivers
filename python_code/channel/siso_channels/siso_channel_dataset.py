from typing import Tuple

import numpy as np
from numpy.random import default_rng

from python_code import conf
from python_code.channel.modulator import MODULATION_DICT
from python_code.channel.siso_channels.isi_awgn_channel import ISIAWGNChannel
from python_code.utils.constants import ChannelModels, ModulationType
from python_code.utils.probs_utils import break_transmitted_siso_word_to_symbols

SISO_CHANNELS_DICT = {ChannelModels.Synthetic.name: ISIAWGNChannel}


class SISOChannel:
    def __init__(self, block_length: int, pilots_length: int, fading_in_channel: bool):
        self._block_length = block_length
        self._pilots_length = pilots_length
        self._bits_generator = default_rng(seed=conf.seed)
        self.h_shape = [1, conf.memory_length]
        self.fading_in_channel = fading_in_channel
        self.tx_length = conf.memory_length
        self.rx_length = 1

    def _transmit(self, h: np.ndarray, snr: float) -> Tuple[np.ndarray, np.ndarray]:
        # create pilots and data
        tx_pilots = self._bits_generator.integers(0, 2, size=(1, self._pilots_length)).reshape(1, -1)
        tx_data = self._bits_generator.integers(0, 2, size=(1, self._block_length - self._pilots_length))
        tx = np.concatenate([tx_pilots, tx_data], axis=1).reshape(1, -1)
        # add zero bits
        padded_tx = np.concatenate(
            [np.zeros([tx.shape[0], conf.memory_length - 1]), tx, np.zeros([tx.shape[0], conf.memory_length])], axis=1)
        if conf.modulation_type == ModulationType.QPSK.name:
            print("Did not implement the QPSK constellation for the SISO case, switch to BPSK or MIMO!")
            raise ValueError("Did not implement the QPSK constellation for the SISO case, switch to BPSK or MIMO!")
        # modulation
        s = MODULATION_DICT[conf.modulation_type].modulate(padded_tx)
        # transmit through noisy channel
        rx = SISO_CHANNELS_DICT[conf.channel_model].transmit(s=s, h=h, snr=snr, memory_length=conf.memory_length)
        symbols, rx = break_transmitted_siso_word_to_symbols(conf.memory_length, tx), rx.T
        return symbols[:-conf.memory_length + 1], rx[:-conf.memory_length + 1]

    def get_vectors(self, snr: float, index: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        # get channel values
        if conf.channel_model == ChannelModels.Synthetic.name:
            h = ISIAWGNChannel.calculate_channel(conf.memory_length, fading=self.fading_in_channel, index=index)
        else:
            raise ValueError("No such channel model!!!")
        tx, rx = self._transmit(h, snr)
        return tx, h, rx
