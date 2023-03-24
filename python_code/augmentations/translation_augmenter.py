from random import randint
from typing import Tuple

import torch

from python_code import DEVICE
from python_code.augmentations.rotation_augmenter import DEG_IN_CIRCLE
from python_code.channel.modulator import MODULATION_NUM_MAPPING
from python_code.utils.config_singleton import Config
from python_code.utils.constants import ChannelModes, ModulationType
from python_code.utils.probs_utils import calculate_mimo_states, calculate_siso_states

conf = Config()

TX_MAPPING_DICT = {
    ModulationType.BPSK.name:
        {0: 1,
         1: 0},
    ModulationType.QPSK.name:
        {0: 1,
         1: 3,
         3: 2,
         2: 0}
}

RX_MAPPING_DICT = {
    ModulationType.BPSK.name:
        {0: [-1],
         1: [-1]},
    ModulationType.QPSK.name:
        {0: [-1, 1],
         1: [1, -1],
         3: [-1, 1],
         2: [1, -1]}
}


class TranslationAugmenter:
    """
    One of the proposed augmentations schemes. Translates a given point to another cluster.
    """

    def __init__(self, centers: torch.Tensor):
        super().__init__()
        self._centers = centers
        self.degrees = list(range(0, DEG_IN_CIRCLE, DEG_IN_CIRCLE // MODULATION_NUM_MAPPING[conf.modulation_type]))

    def augment(self, rx: torch.Tensor, tx: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        if conf.channel_type == ChannelModes.SISO.name:
            received_word_state = calculate_siso_states(conf.memory_length, tx)[0]
        elif conf.channel_type == ChannelModes.MIMO.name:
            received_word_state = calculate_mimo_states(conf.n_user, tx.reshape(1, -1))[0]
            rx = rx[0]
        else:
            raise ValueError("No such channel type!!!")
        # choose the new cluster / class randomly
        random_ind = randint(a=1, b=len(self.degrees) - 1)
        new_tx = tx[0]
        tx_map = TX_MAPPING_DICT[conf.modulation_type]
        rx_map = RX_MAPPING_DICT[conf.modulation_type]
        rx_transformation = torch.ones(rx.shape).to(DEVICE)
        # apply the transformations to get the new transformed tx and rx
        for i in range(random_ind):
            rx_transformation *= torch.tensor([rx_map[x.item()] for x in new_tx])[:rx.shape[0]].reshape(
                rx.shape).to(DEVICE)
            new_tx = torch.tensor([tx_map[x.item()] for x in new_tx]).to(DEVICE)
        if conf.channel_type == ChannelModes.SISO.name:
            new_state = calculate_siso_states(conf.memory_length, new_tx)[0]
        elif conf.channel_type == ChannelModes.MIMO.name:
            new_state = calculate_mimo_states(conf.n_user, new_tx.reshape(1, -1))[0]
        else:
            raise ValueError("No such channel type!!!")
        if torch.count_nonzero(self._centers[new_state.item()]) > 0:
            # apply the transformation to rx to get the new transformed rx, check out the paper for more details
            transformed_received = rx_transformation * rx
            delta = self._centers[new_state.item()] - rx_transformation * self._centers[received_word_state.item()]
            new_rx = delta + transformed_received
            if conf.channel_type == ChannelModes.MIMO.name:
                new_rx = new_rx.unsqueeze(0)
            new_tx = new_tx.unsqueeze(0)
            return new_rx, new_tx
        if conf.channel_type == ChannelModes.MIMO.name:
            rx = rx.unsqueeze(0)
        return rx, tx

    @property
    def centers(self) -> torch.Tensor:
        return self._centers
