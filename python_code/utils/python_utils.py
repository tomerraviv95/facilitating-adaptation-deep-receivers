import copy
import math
import pickle as pkl
from typing import Dict, Any, List

import numpy as np
from torch import nn

from python_code import conf
from python_code.channel.modulator import MODULATION_NUM_MAPPING


def save_pkl(pkls_path: str, array: np.ndarray, type: str):
    output = open(pkls_path + '_' + type + '.pkl', 'wb')
    pkl.dump(array, output)
    output.close()


def load_pkl(pkls_path: str, type: str) -> Dict[Any, Any]:
    output = open(pkls_path + '_' + type + '.pkl', 'rb')
    return pkl.load(output)


def normalize_for_modulation(size: int) -> int:
    """
    Return the bits/symbols ratio for the given block size
    """
    return int(size // int(math.log2(MODULATION_NUM_MAPPING[conf.modulation_type])))


def copy_model(model: List[nn.Module]) -> List[nn.Module]:
    return [copy.deepcopy(single_model) for single_model in model]
