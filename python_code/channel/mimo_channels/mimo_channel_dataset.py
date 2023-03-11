from typing import Tuple

import numpy as np
from numpy.random import default_rng

from python_code import conf
from python_code.channel.mimo_channels.cost_channel import Cost2100MIMOChannel
from python_code.channel.mimo_channels.sed_channel import SEDChannel
from python_code.channel.modulator import MODULATION_DICT
from python_code.utils.constants import ModulationType, ChannelModels
from python_code.utils.probs_utils import get_qpsk_symbols_from_bits

MIMO_CHANNELS_DICT = {ChannelModels.Synthetic.name: SEDChannel,
                      ChannelModels.Cost2100.name: Cost2100MIMOChannel}


class MIMOChannel:
    def __init__(self, block_length: int, pilots_length: int):
        self._block_length = block_length
        self._pilots_length = pilots_length
        self._bits_generator = default_rng(seed=conf.seed)
        self.tx_length = conf.n_user
        self.h_shape = [conf.n_ant, conf.n_user]
        self.rx_length = conf.n_ant

    def _transmit(self, h: np.ndarray, snr: float) -> Tuple[np.ndarray, np.ndarray]:
        tx_pilots = self._bits_generator.integers(0, 2, size=(self._pilots_length, conf.n_user))
        tx_data = self._bits_generator.integers(0, 2, size=(self._block_length - self._pilots_length, conf.n_user))
        tx = np.concatenate([tx_pilots, tx_data])
        # modulation
        s = MODULATION_DICT[conf.modulation_type].modulate(tx.T)
        # pass through channel
        rx = MIMO_CHANNELS_DICT[conf.channel_model].transmit(s=s, h=h, snr=snr)
        if conf.modulation_type == ModulationType.QPSK.name:
            tx = get_qpsk_symbols_from_bits(tx)
        return tx, rx.T

    def get_vectors(self, snr: float, index: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        # get channel values
        if conf.channel_model == ChannelModels.Synthetic.name:
            h = SEDChannel.calculate_channel(conf.n_ant, conf.n_user, index, conf.fading_in_channel)
        elif conf.channel_model == ChannelModels.Cost2100.name:
            h = Cost2100MIMOChannel.calculate_channel(conf.n_ant, conf.n_user, index, conf.fading_in_channel)
        else:
            raise ValueError("No such channel model!!!")
        tx, rx = self._transmit(h, snr)
        return tx, h, rx
