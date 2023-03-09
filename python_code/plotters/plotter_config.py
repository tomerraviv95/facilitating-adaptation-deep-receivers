from enum import Enum
from typing import Tuple, List, Dict

from python_code.utils.constants import DetectorType


class PlotType(Enum):
    COMPARE_BLACK_BOX_AND_MODEL_BASED = 'COMPARE_BLACK_BOX_AND_MODEL_BASED'
    LINEAR_SYNTH_QPSK = 'LINEAR_SYNTH_QPSK'
    NON_LINEAR_SYNTH_QPSK = 'NON_LINEAR_SYNTH_QPSK'


def get_config(plot_type: PlotType) -> Tuple[List[Dict], list, str, str]:
    if plot_type == PlotType.COMPARE_BLACK_BOX_AND_MODEL_BASED:
        params_dicts = [
            {'snr': 4, 'detector_type': DetectorType.black_box.name},
            {'snr': 6, 'detector_type': DetectorType.black_box.name},
            {'snr': 8, 'detector_type': DetectorType.black_box.name},
            {'snr': 10, 'detector_type': DetectorType.black_box.name},
            {'snr': 12, 'detector_type': DetectorType.black_box.name},
            {'snr': 4, 'detector_type': DetectorType.deepsic.name},
            {'snr': 6, 'detector_type': DetectorType.deepsic.name},
            {'snr': 8, 'detector_type': DetectorType.deepsic.name},
            {'snr': 10, 'detector_type': DetectorType.deepsic.name},
            {'snr': 12, 'detector_type': DetectorType.deepsic.name},
        ]
        values = list(range(4, 13, 2))
        xlabel, ylabel = 'SNR', 'SER'
    elif plot_type == PlotType.LINEAR_SYNTH_QPSK:
        params_dicts = [
            {'snr': 10, 'detector_type': DetectorType.black_box.name},
            {'snr': 11, 'detector_type': DetectorType.black_box.name},
            {'snr': 12, 'detector_type': DetectorType.black_box.name},
            {'snr': 13, 'detector_type': DetectorType.black_box.name},
            {'snr': 14, 'detector_type': DetectorType.black_box.name},
            {'snr': 10, 'detector_type': DetectorType.deepsic.name},
            {'snr': 11, 'detector_type': DetectorType.deepsic.name},
            {'snr': 12, 'detector_type': DetectorType.deepsic.name},
            {'snr': 13, 'detector_type': DetectorType.deepsic.name},
            {'snr': 14, 'detector_type': DetectorType.deepsic.name},
            {'snr': 10, 'detector_type': DetectorType.deepsic.name,
             'aug_type': ['geometric_augmenter', 'translation_augmenter', 'rotation_augmenter'], 'online_repeats_n': 3},
            {'snr': 11, 'detector_type': DetectorType.deepsic.name,
             'aug_type': ['geometric_augmenter', 'translation_augmenter', 'rotation_augmenter'], 'online_repeats_n': 3},
            {'snr': 12, 'detector_type': DetectorType.deepsic.name,
             'aug_type': ['geometric_augmenter', 'translation_augmenter', 'rotation_augmenter'], 'online_repeats_n': 3},
            {'snr': 13, 'detector_type': DetectorType.deepsic.name,
             'aug_type': ['geometric_augmenter', 'translation_augmenter', 'rotation_augmenter'], 'online_repeats_n': 3},
            {'snr': 14, 'detector_type': DetectorType.deepsic.name,
             'aug_type': ['geometric_augmenter', 'translation_augmenter', 'rotation_augmenter'], 'online_repeats_n': 3},
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
        ]
        values = list(range(10, 15, 1))
        xlabel, ylabel = 'SNR', 'SER'
    elif plot_type == PlotType.NON_LINEAR_SYNTH_QPSK:
        params_dicts = [
            {'snr': 10, 'detector_type': DetectorType.black_box.name},
            {'snr': 11, 'detector_type': DetectorType.black_box.name},
            {'snr': 12, 'detector_type': DetectorType.black_box.name},
            {'snr': 13, 'detector_type': DetectorType.black_box.name},
            {'snr': 14, 'detector_type': DetectorType.black_box.name},
            {'snr': 10, 'detector_type': DetectorType.deepsic.name},
            {'snr': 11, 'detector_type': DetectorType.deepsic.name},
            {'snr': 12, 'detector_type': DetectorType.deepsic.name},
            {'snr': 13, 'detector_type': DetectorType.deepsic.name},
            {'snr': 14, 'detector_type': DetectorType.deepsic.name},
            {'snr': 10, 'detector_type': DetectorType.deepsic.name,
             'aug_type': ['geometric_augmenter', 'translation_augmenter', 'rotation_augmenter'], 'online_repeats_n': 3},
            {'snr': 11, 'detector_type': DetectorType.deepsic.name,
             'aug_type': ['geometric_augmenter', 'translation_augmenter', 'rotation_augmenter'], 'online_repeats_n': 3},
            {'snr': 12, 'detector_type': DetectorType.deepsic.name,
             'aug_type': ['geometric_augmenter', 'translation_augmenter', 'rotation_augmenter'], 'online_repeats_n': 3},
            {'snr': 13, 'detector_type': DetectorType.deepsic.name,
             'aug_type': ['geometric_augmenter', 'translation_augmenter', 'rotation_augmenter'], 'online_repeats_n': 3},
            {'snr': 14, 'detector_type': DetectorType.deepsic.name,
             'aug_type': ['geometric_augmenter', 'translation_augmenter', 'rotation_augmenter'], 'online_repeats_n': 3},
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
        ]
        values = list(range(10, 15, 1))
        xlabel, ylabel = 'SNR', 'SER'
    else:
        raise ValueError('No such plot type!!!')

    return params_dicts, values, xlabel, ylabel
