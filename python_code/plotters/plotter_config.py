from enum import Enum
from typing import Tuple, List, Dict

from python_code.utils.constants import DetectorType


class PlotType(Enum):
    LINEAR_SYNTH_QPSK = 'LINEAR_SYNTH_QPSK'
    SISO = 'SISO'


def get_config(plot_type: PlotType) -> Tuple[List[Dict], list, str, str]:
    if plot_type == PlotType.LINEAR_SYNTH_QPSK:
        params_dicts = [
            {'snr': 10, 'detector_type': DetectorType.joint_black_box.name},
            {'snr': 11, 'detector_type': DetectorType.joint_black_box.name},
            {'snr': 12, 'detector_type': DetectorType.joint_black_box.name},
            {'snr': 13, 'detector_type': DetectorType.joint_black_box.name},
            {'snr': 14, 'detector_type': DetectorType.joint_black_box.name},
            {'snr': 15, 'detector_type': DetectorType.joint_black_box.name},
            {'snr': 16, 'detector_type': DetectorType.joint_black_box.name},
            {'snr': 10, 'detector_type': DetectorType.online_black_box.name},
            {'snr': 11, 'detector_type': DetectorType.online_black_box.name},
            {'snr': 12, 'detector_type': DetectorType.online_black_box.name},
            {'snr': 13, 'detector_type': DetectorType.online_black_box.name},
            {'snr': 14, 'detector_type': DetectorType.online_black_box.name},
            {'snr': 15, 'detector_type': DetectorType.online_black_box.name},
            {'snr': 16, 'detector_type': DetectorType.online_black_box.name},
            {'snr': 10, 'detector_type': DetectorType.joint_deepsic.name},
            {'snr': 11, 'detector_type': DetectorType.joint_deepsic.name},
            {'snr': 12, 'detector_type': DetectorType.joint_deepsic.name},
            {'snr': 13, 'detector_type': DetectorType.joint_deepsic.name},
            {'snr': 14, 'detector_type': DetectorType.joint_deepsic.name},
            {'snr': 15, 'detector_type': DetectorType.joint_deepsic.name},
            {'snr': 16, 'detector_type': DetectorType.joint_deepsic.name},
            {'snr': 10, 'detector_type': DetectorType.online_deepsic.name},
            {'snr': 11, 'detector_type': DetectorType.online_deepsic.name},
            {'snr': 12, 'detector_type': DetectorType.online_deepsic.name},
            {'snr': 13, 'detector_type': DetectorType.online_deepsic.name},
            {'snr': 14, 'detector_type': DetectorType.online_deepsic.name},
            {'snr': 15, 'detector_type': DetectorType.online_deepsic.name},
            {'snr': 16, 'detector_type': DetectorType.online_deepsic.name},
            {'snr': 10, 'detector_type': DetectorType.meta_deepsic.name},
            {'snr': 11, 'detector_type': DetectorType.meta_deepsic.name},
            {'snr': 12, 'detector_type': DetectorType.meta_deepsic.name},
            {'snr': 13, 'detector_type': DetectorType.meta_deepsic.name},
            {'snr': 14, 'detector_type': DetectorType.meta_deepsic.name},
            {'snr': 15, 'detector_type': DetectorType.meta_deepsic.name},
            {'snr': 16, 'detector_type': DetectorType.meta_deepsic.name},
            {'snr': 10, 'detector_type': DetectorType.meta_deepsic.name,
             'aug_type': ['geometric_augmenter', 'translation_augmenter', 'rotation_augmenter'], 'online_repeats_n': 3},
            {'snr': 11, 'detector_type': DetectorType.meta_deepsic.name,
             'aug_type': ['geometric_augmenter', 'translation_augmenter', 'rotation_augmenter'], 'online_repeats_n': 3},
            {'snr': 12, 'detector_type': DetectorType.meta_deepsic.name,
             'aug_type': ['geometric_augmenter', 'translation_augmenter', 'rotation_augmenter'], 'online_repeats_n': 3},
            {'snr': 13, 'detector_type': DetectorType.meta_deepsic.name,
             'aug_type': ['geometric_augmenter', 'translation_augmenter', 'rotation_augmenter'], 'online_repeats_n': 3},
            {'snr': 14, 'detector_type': DetectorType.meta_deepsic.name,
             'aug_type': ['geometric_augmenter', 'translation_augmenter', 'rotation_augmenter'], 'online_repeats_n': 3},
            {'snr': 15, 'detector_type': DetectorType.meta_deepsic.name,
             'aug_type': ['geometric_augmenter', 'translation_augmenter', 'rotation_augmenter'], 'online_repeats_n': 3},
            {'snr': 16, 'detector_type': DetectorType.meta_deepsic.name,
             'aug_type': ['geometric_augmenter', 'translation_augmenter', 'rotation_augmenter'], 'online_repeats_n': 3},
        ]
        values = list(range(10, 17, 1))
        xlabel, ylabel = 'SNR [dB]', 'BER'
    elif plot_type == PlotType.SISO:
        params_dicts = [
            {'snr': 10, 'detector_type': DetectorType.joint_rnn.name},
            {'snr': 11, 'detector_type': DetectorType.joint_rnn.name},
            {'snr': 12, 'detector_type': DetectorType.joint_rnn.name},
            {'snr': 13, 'detector_type': DetectorType.joint_rnn.name},
            {'snr': 14, 'detector_type': DetectorType.joint_rnn.name},
            {'snr': 15, 'detector_type': DetectorType.joint_rnn.name},
            {'snr': 16, 'detector_type': DetectorType.joint_rnn.name},
            {'snr': 10, 'detector_type': DetectorType.online_rnn.name},
            {'snr': 11, 'detector_type': DetectorType.online_rnn.name},
            {'snr': 12, 'detector_type': DetectorType.online_rnn.name},
            {'snr': 13, 'detector_type': DetectorType.online_rnn.name},
            {'snr': 14, 'detector_type': DetectorType.online_rnn.name},
            {'snr': 15, 'detector_type': DetectorType.online_rnn.name},
            {'snr': 16, 'detector_type': DetectorType.online_rnn.name},
            {'snr': 10, 'detector_type': DetectorType.joint_viterbinet.name},
            {'snr': 11, 'detector_type': DetectorType.joint_viterbinet.name},
            {'snr': 12, 'detector_type': DetectorType.joint_viterbinet.name},
            {'snr': 13, 'detector_type': DetectorType.joint_viterbinet.name},
            {'snr': 14, 'detector_type': DetectorType.joint_viterbinet.name},
            {'snr': 15, 'detector_type': DetectorType.joint_viterbinet.name},
            {'snr': 16, 'detector_type': DetectorType.joint_viterbinet.name},
            {'snr': 10, 'detector_type': DetectorType.online_viterbinet.name},
            {'snr': 11, 'detector_type': DetectorType.online_viterbinet.name},
            {'snr': 12, 'detector_type': DetectorType.online_viterbinet.name},
            {'snr': 13, 'detector_type': DetectorType.online_viterbinet.name},
            {'snr': 14, 'detector_type': DetectorType.online_viterbinet.name},
            {'snr': 15, 'detector_type': DetectorType.online_viterbinet.name},
            {'snr': 16, 'detector_type': DetectorType.online_viterbinet.name},
            {'snr': 10, 'detector_type': DetectorType.meta_viterbinet.name},
            {'snr': 11, 'detector_type': DetectorType.meta_viterbinet.name},
            {'snr': 12, 'detector_type': DetectorType.meta_viterbinet.name},
            {'snr': 13, 'detector_type': DetectorType.meta_viterbinet.name},
            {'snr': 14, 'detector_type': DetectorType.meta_viterbinet.name},
            {'snr': 15, 'detector_type': DetectorType.meta_viterbinet.name},
            {'snr': 16, 'detector_type': DetectorType.meta_viterbinet.name},
            {'snr': 10, 'detector_type': DetectorType.meta_viterbinet.name,
             'aug_type': ['geometric_augmenter', 'translation_augmenter', 'rotation_augmenter'], 'online_repeats_n': 3},
            {'snr': 11, 'detector_type': DetectorType.meta_viterbinet.name,
             'aug_type': ['geometric_augmenter', 'translation_augmenter', 'rotation_augmenter'], 'online_repeats_n': 3},
            {'snr': 12, 'detector_type': DetectorType.meta_viterbinet.name,
             'aug_type': ['geometric_augmenter', 'translation_augmenter', 'rotation_augmenter'], 'online_repeats_n': 3},
            {'snr': 13, 'detector_type': DetectorType.meta_viterbinet.name,
             'aug_type': ['geometric_augmenter', 'translation_augmenter', 'rotation_augmenter'], 'online_repeats_n': 3},
            {'snr': 14, 'detector_type': DetectorType.meta_viterbinet.name,
             'aug_type': ['geometric_augmenter', 'translation_augmenter', 'rotation_augmenter'], 'online_repeats_n': 3},
            {'snr': 15, 'detector_type': DetectorType.meta_viterbinet.name,
             'aug_type': ['geometric_augmenter', 'translation_augmenter', 'rotation_augmenter'], 'online_repeats_n': 3},
            {'snr': 16, 'detector_type': DetectorType.meta_viterbinet.name,
             'aug_type': ['geometric_augmenter', 'translation_augmenter', 'rotation_augmenter'], 'online_repeats_n': 3},
        ]
        values = list(range(10, 17, 1))
        xlabel, ylabel = 'SNR [dB]', 'BER'
    else:
        raise ValueError('No such plot type!!!')

    return params_dicts, values, xlabel, ylabel
