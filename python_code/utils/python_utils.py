import math
import pickle as pkl
from typing import Dict, Any

import numpy as np

from python_code import conf
from python_code.channel.modulator import MODULATION_NUM_MAPPING


def save_pkl(pkls_path: str, array: np.ndarray):
    output = open(pkls_path + '.pkl', 'wb')
    pkl.dump(array, output)
    output.close()


def load_pkl(pkls_path: str) -> Dict[Any, Any]:
    output = open(pkls_path + '.pkl', 'rb')
    return pkl.load(output)


def normalize_for_modulation(size: int) -> int:
    """
    Return the bits/symbols ratio for the given block size
    """
    return int(size // int(math.log2(MODULATION_NUM_MAPPING[conf.modulation_type])))
