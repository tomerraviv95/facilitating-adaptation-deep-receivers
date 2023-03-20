from enum import Enum

HALF = 0.5


class Phase(Enum):
    TRAIN = 'train'
    TEST = 'test'


class ChannelModes(Enum):
    MIMO = 'MIMO'


class ChannelModels(Enum):
    Synthetic = 'Synthetic'
    Cost2100 = 'Cost2100'


class DetectorType(Enum):
    joint_black_box = 'joint_black_box'
    online_black_box = 'online_black_box'
    joint_deepsic = 'joint_deepsic'
    online_deepsic = 'online_deepsic'
    meta_deepsic = 'meta_deepsic'


class ModulationType(Enum):
    BPSK = 'BPSK'
    QPSK = 'QPSK'
