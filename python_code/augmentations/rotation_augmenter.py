import math
from random import randint
from typing import Tuple

import torch

from python_code import DEVICE
from python_code.channel.modulator import MODULATION_NUM_MAPPING
from python_code.utils.config_singleton import Config
from python_code.utils.constants import ModulationType

conf = Config()

DEG_IN_CIRCLE = 360

MAPPING_DICT = {
    ModulationType.BPSK.name:
        {0: 1,
         1: 0},
    ModulationType.QPSK.name:
        {0: 1,
         1: 3,
         3: 2,
         2: 0}
}


class RotationAugmenter:
    """
    One of the proposed augmentations scheme. Rotate the constellation by a constellation-conserving projection.
    """

    def __init__(self):
        ## creating the rotation-preserving degrees
        deg_list = list(range(0, DEG_IN_CIRCLE, DEG_IN_CIRCLE // MODULATION_NUM_MAPPING[conf.modulation_type]))
        rad_list = [math.radians(degree) for degree in deg_list]
        self.degrees = torch.Tensor(rad_list).to(DEVICE)

    def augment(self, rx: torch.Tensor, tx: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        random_ind = randint(a=1, b=len(self.degrees) - 1)
        # choose a random degree
        chosen_transformation = self.degrees[random_ind]
        if conf.modulation_type == ModulationType.BPSK.name:
            rx = torch.cat([rx.unsqueeze(-1), torch.zeros_like(rx.unsqueeze(-1))],
                           dim=1)
            rx = rx[:, :, 0]
        # add the random degree to the angle of current word
        new_angle = torch.view_as_complex(rx).angle() + chosen_transformation
        new_complex_rx = torch.view_as_complex(rx).abs() * (torch.cos(new_angle) + 1j * torch.sin(new_angle))
        new_rx = torch.view_as_real(new_complex_rx)
        if conf.modulation_type == ModulationType.BPSK.name:
            new_rx = new_rx[:, 0].unsqueeze(1)

        # get the desired new class after transformation
        new_tx = tx
        map = MAPPING_DICT[conf.modulation_type]
        for i in range(random_ind):
            new_tx = torch.tensor([map[x.item()] for x in new_tx[0]]).reshape(tx.shape)
        return new_rx, new_tx.to(DEVICE)
