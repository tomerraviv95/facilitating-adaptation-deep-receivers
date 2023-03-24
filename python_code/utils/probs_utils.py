import itertools

import numpy as np
import torch

from python_code import DEVICE, conf
from python_code.channel.modulator import MODULATION_NUM_MAPPING
from python_code.utils.constants import HALF, ModulationType


def generate_bits_by_state(state: int, n_state: int) -> torch.Tensor:
    """
    Calculates all possible combinations of vector of length state, with elements from 0,..,n_states-1
    Given by the constellation size.
    """
    combinations = list(itertools.product(range(MODULATION_NUM_MAPPING[conf.modulation_type]), repeat=n_state))
    return torch.Tensor(combinations[state][::-1]).reshape(1, n_state).to(DEVICE)

def calculate_siso_states(memory_length: int, transmitted_words: torch.Tensor) -> torch.Tensor:
    """
    calculates siso states vector for the transmitted words. Number of states is 2 ** memory length.
    Only BPSK is allowed.
    :param memory_length: length of channel memory
    :param transmitted_words: channel transmitted words
    :return: vector of length of transmitted_words with values in the range of 0,1,...,n_states-1
    """
    states_enumerator = (2 ** torch.arange(memory_length)).reshape(1, -1).float().to(DEVICE)
    gt_states = torch.sum(transmitted_words * states_enumerator, dim=1).long()
    return gt_states


def calculate_mimo_states(n_user: int, transmitted_words: torch.Tensor) -> torch.Tensor:
    """
    Calculates mimo states vector for the transmitted words. Number of states is 2/4/8 ** n_user depending on the
    constellation size.
    """
    states_enumerator = (MODULATION_NUM_MAPPING[conf.modulation_type] ** torch.arange(n_user)).to(DEVICE)
    gt_states = torch.sum(transmitted_words * states_enumerator, dim=1).long()
    return gt_states

def break_transmitted_siso_word_to_symbols(memory_length: int, transmitted_words: np.ndarray) -> np.ndarray:
    """
    Take words of bits b_0,..b_(n-1) with length n, and creates [memory_length X n-memory_length] matrix
    with bits b_0,..., b_(n-1-memory_length) at first row, bits b_1,..., b_(n-1-memory_length+1) at second row
    up to memory_length such rows
    """
    padded = np.concatenate([np.zeros([transmitted_words.shape[0], memory_length - 1]), transmitted_words,
                             np.zeros([transmitted_words.shape[0], memory_length])], axis=1)
    unsqueezed_padded = np.expand_dims(padded, axis=1)
    blockwise_words = np.concatenate([unsqueezed_padded[:, :, i:-memory_length + i] for i in range(memory_length)],
                                     axis=1)
    return blockwise_words.squeeze().T



def calculate_symbols_from_states(state_size: int, gt_states: torch.Tensor) -> torch.Tensor:
    """
    Used for the dnn-aided receivers. Calculates the symbols from the states.
    """
    mask = MODULATION_NUM_MAPPING[conf.modulation_type] ** torch.arange(state_size).to(DEVICE, gt_states.dtype)
    if conf.modulation_type == ModulationType.BPSK.name:
        return gt_states.unsqueeze(-1).bitwise_and(mask).ne(0).long()
    elif conf.modulation_type in [ModulationType.QPSK.name]:
        result = torch.div(gt_states.unsqueeze(-1), mask, rounding_mode='floor')
        result = result % MODULATION_NUM_MAPPING[conf.modulation_type]
        return result


def prob_to_BPSK_symbol(p: torch.Tensor) -> torch.Tensor:
    """
    Converts Probabilities to BPSK Symbols by hard threshold: [0,0.5] -> '-1', [0.5,1] -> '+1'
    """
    return torch.sign(p - HALF)


def prob_to_QPSK_symbol(p: torch.Tensor) -> torch.Tensor:
    """
    Converts Probabilities to QPSK Symbols by hard threshold.
    first bit: [0,0.5] -> '+1',[0.5,1] -> '-1'
    second bit: [0,0.5] -> '+1',[0.5,1] -> '-1'
    """
    p_real_neg = p[:, :, 0] + p[:, :, 2]
    first_symbol = (-1) * torch.sign(p_real_neg - HALF)
    p_img_neg = p[:, :, 1] + p[:, :, 2]
    second_symbol = (-1) * torch.sign(p_img_neg - HALF)
    s = torch.cat([first_symbol.unsqueeze(-1), second_symbol.unsqueeze(-1)], dim=-1)
    return torch.view_as_complex(s)


def get_qpsk_symbols_from_bits(b: np.ndarray) -> np.ndarray:
    return b[::2] + 2 * b[1::2]


def get_bits_from_qpsk_symbols(target: torch.Tensor) -> torch.Tensor:
    first_bit = target % 2
    second_bit = torch.floor(target / 2) % 2
    target = torch.cat([first_bit.unsqueeze(-1), second_bit.unsqueeze(-1)], dim=2).transpose(1, 2).reshape(
        2 * first_bit.shape[0], -1)
    return target
