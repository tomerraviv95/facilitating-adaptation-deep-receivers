from enum import Enum

HALF = 0.5


class Phase(Enum):
    TRAIN = 'train'
    TEST = 'test'


class ChannelModes(Enum):
    MIMO = 'MIMO'


class ChannelModels(Enum):
    Synthetic = 'Synthetic'


class DetectorType(Enum):
    deepsic = 'deepsic'
    meta_deepsic = 'meta_deepsic'
    black_box = 'black_box'


class ModulationType(Enum):
    BPSK = 'BPSK'
    QPSK = 'QPSK'
