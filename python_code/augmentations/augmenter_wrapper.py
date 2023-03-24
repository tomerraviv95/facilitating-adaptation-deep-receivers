from typing import Tuple, List

import torch

from python_code import DEVICE
from python_code.augmentations.geometric_augmenter import GeometricAugmenter
from python_code.augmentations.no_sampler import NoSampler
from python_code.augmentations.rotation_augmenter import RotationAugmenter
from python_code.augmentations.translation_augmenter import TranslationAugmenter
from python_code.channel.modulator import MODULATION_NUM_MAPPING
from python_code.utils.config_singleton import Config
from python_code.utils.constants import ChannelModes, ModulationType
from python_code.utils.probs_utils import calculate_mimo_states, calculate_siso_states

conf = Config()

MIN_CLUSTER_POINTS = 80


def estimate_params(rx: torch.Tensor, tx: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, int, int]:
    """
    Estimate parameters of centers and stds in the jth step based on the known states of the pilots.
    :param rx: received pilots word
    :param tx: transmitted pilots word
    :return: updated centers and stds values per class
    """
    if conf.channel_type == ChannelModes.SISO.name:
        gt_states = calculate_siso_states(conf.memory_length, tx)
        n_states = MODULATION_NUM_MAPPING[conf.modulation_type] ** conf.memory_length
        state_size = 1
    elif conf.channel_type == ChannelModes.MIMO.name:
        gt_states = calculate_mimo_states(conf.n_user, tx)
        n_states = MODULATION_NUM_MAPPING[conf.modulation_type] ** conf.n_user
        state_size = conf.n_ant
    else:
        raise ValueError("No such channel type!!!")

    centers = torch.empty([n_states, *rx.shape[1:]]).to(DEVICE)
    stds = torch.empty([n_states, *rx.shape[1:]]).to(DEVICE)

    for state in range(n_states):
        state_ind = (gt_states == state)
        if torch.count_nonzero(state_ind) > MIN_CLUSTER_POINTS:
            state_received = rx[state_ind]
            stds[state] = torch.std(state_received, dim=0)
            centers[state] = torch.mean(state_received.real, dim=0)
        else:
            centers[state] = 0
    stds[torch.isnan(stds)] = torch.mean(stds[~torch.isnan(stds)])
    return centers, stds, gt_states, n_states, state_size


ALPHA1 = 0.3
ALPHA2 = 0.3


class AugmenterWrapper:

    def __init__(self, augmentations: List[str], fading_in_channel: bool):
        self._augmentations = augmentations
        self._fading_in_channel = fading_in_channel
        self._centers = None
        self._stds = None
        self.active_augmentations_num = max(len(self._augmentations), 1)

    def update_hyperparams(self, received_words: torch.Tensor, transmitted_words: torch.Tensor):
        if conf.modulation_type == ModulationType.QPSK.name:
            received_words = torch.view_as_real(received_words)
        centers, stds, gt_states, n_states, state_size = estimate_params(received_words, transmitted_words)

        if self._fading_in_channel:
            self._centers, self._stds = self.smooth_parameters(centers, stds)
        else:
            self._centers, self._stds = centers, stds

        self._sampler = NoSampler(received_words, transmitted_words)

        self._augmenters_dict = {
            'rotation_augmenter': RotationAugmenter(),
            'translation_augmenter': TranslationAugmenter(self._centers),
            'geometric_augmenter': GeometricAugmenter(self._centers, self._stds, n_states, state_size, gt_states),
        }

        self._n_states = n_states

    def smooth_parameters(self, cur_centers: torch.Tensor, cur_stds: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Update the parameters via temporal smoothing over a window with parameter alpha
        :param cur_centers: jth step estimated centers
        :param cur_stds:  jth step estimated stds
        :return: smoothed centers and stds vectors
        """

        # self._centers = cur_centers
        if self._centers is not None:
            centers = ALPHA1 * cur_centers + (1 - ALPHA1) * self._centers
        else:
            centers = cur_centers

        if self._stds is not None:
            stds = ALPHA2 * cur_stds + (1 - ALPHA2) * self._stds
        else:
            stds = cur_stds

        return centers, stds

    @property
    def n_states(self) -> int:
        return self._n_states

    def augment_single(self, i: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Augment the received word using one of the given augmentations methods.
        :param i: current repetition index
        :return: the augmented received and transmitted pairs
        """
        aug_rxs, aug_txs = [], []
        # sample via the sampling method
        rx, tx = self._sampler.sample(i)
        if len(self._augmentations) == 0:
            return rx, tx

        # run through the desired augmentations
        for augmentation_name in self._augmentations:
            augmenter = self._augmenters_dict[augmentation_name]
            aug_rx, aug_tx = augmenter.augment(rx.clone(), tx.clone())
            aug_rxs.append(aug_rx), aug_txs.append(aug_tx)

        reshaped_aug_txs = torch.cat(aug_txs).to(DEVICE).reshape(self.active_augmentations_num, -1)
        if conf.modulation_type == ModulationType.QPSK.name:
            return torch.cat(aug_rxs).to(DEVICE).reshape(self.active_augmentations_num, -1, 2), reshaped_aug_txs
        else:
            return torch.cat(aug_rxs).to(DEVICE).reshape(self.active_augmentations_num, -1), reshaped_aug_txs

    def augment_batch(self, rx: torch.Tensor, tx: torch.Tensor):
        """
        The main augmentation function, used to augment each pilot in the evaluation phase.
        :param h: channel coefficients
        :param rx: received word
        :param tx: transmitted word
        :return: the augmented batch of (rx,tx)
        """
        aug_tx = torch.empty([(1 + self.active_augmentations_num * conf.online_repeats_n) * tx.shape[0],
                              tx.shape[1]]).to(DEVICE)
        if conf.modulation_type == ModulationType.QPSK.name:
            rx = torch.view_as_real(rx)
        aug_rx = torch.empty(
            [(1 + self.active_augmentations_num * conf.online_repeats_n) * rx.shape[0], *rx.shape[1:]],
            dtype=rx.dtype).to(DEVICE)
        i = 0
        while i < aug_rx.shape[0]:
            # copy |Q| first samples into Q*
            if i < rx.shape[0]:
                aug_rx[i], aug_tx[i] = rx[i], tx[i]
                i += 1
            # synthesize the rest of samples in Q*
            else:
                cur_aug_rx, cur_aug_tx = self.augment_single(i)
                # if SISO, order of samples matters. so place only 1 sample out randomly of the active augmentations.
                if conf.channel_type == ChannelModes.SISO.name:
                    j = (i // tx.shape[0]) % self.active_augmentations_num
                    aug_rx[i] = cur_aug_rx[j]
                    aug_tx[i] = cur_aug_tx[j]
                    i += 1
                else:
                    aug_rx[i:i + self.active_augmentations_num] = cur_aug_rx
                    aug_tx[i:i + self.active_augmentations_num] = cur_aug_tx
                    i += self.active_augmentations_num

        if conf.modulation_type == ModulationType.QPSK.name:
            aug_rx = torch.view_as_complex(aug_rx)
        return aug_rx, aug_tx
