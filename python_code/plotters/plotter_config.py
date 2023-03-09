from enum import Enum
from typing import Tuple, List, Dict

import numpy as np

from python_code.utils.constants import DetectorType


class PlotType(Enum):
    ## The Three Figures for the Paper
    MIMO_BY_SNR_QPSK = 'MIMO_BY_SNR_QPSK'
    MIMO_BY_SNR_EightPSK = 'MIMO_BY_SNR_EightPSK'
    MIMO_BY_RELIABILITY_EightPSK = 'MIMO_BY_RELIABILITY_EightPSK'


def get_config(plot_type: PlotType) -> Tuple[List[Dict], list, str, str]:
    if plot_type == PlotType.MIMO_BY_SNR_QPSK:
        params_dicts = [
            {'snr': 4, 'detector_type': DetectorType.black_box.name},
            {'snr': 6, 'detector_type': DetectorType.black_box.name},
            {'snr': 8, 'detector_type': DetectorType.black_box.name},
            {'snr': 10, 'detector_type': DetectorType.black_box.name},
            {'snr': 12, 'detector_type': DetectorType.black_box.name},
            {'snr': 4, 'detector_type': DetectorType.bayesian.name},
            {'snr': 6, 'detector_type': DetectorType.bayesian.name},
            {'snr': 8, 'detector_type': DetectorType.bayesian.name},
            {'snr': 10, 'detector_type': DetectorType.bayesian.name},
            {'snr': 12, 'detector_type': DetectorType.bayesian.name},
            {'snr': 4, 'detector_type': DetectorType.deepsic.name},
            {'snr': 6, 'detector_type': DetectorType.deepsic.name},
            {'snr': 8, 'detector_type': DetectorType.deepsic.name},
            {'snr': 10, 'detector_type': DetectorType.deepsic.name},
            {'snr': 12, 'detector_type': DetectorType.deepsic.name},
            {'snr': 4, 'detector_type': DetectorType.model_based_bayesian.name},
            {'snr': 6, 'detector_type': DetectorType.model_based_bayesian.name},
            {'snr': 8, 'detector_type': DetectorType.model_based_bayesian.name},
            {'snr': 10, 'detector_type': DetectorType.model_based_bayesian.name},
            {'snr': 12, 'detector_type': DetectorType.model_based_bayesian.name},
        ]
        values = list(range(4, 13, 2))
        xlabel, ylabel = 'SNR', 'SER'
    elif plot_type == PlotType.MIMO_BY_SNR_EightPSK:
        params_dicts = [
            {'snr': 4, 'detector_type': DetectorType.black_box.name},
            {'snr': 6, 'detector_type': DetectorType.black_box.name},
            {'snr': 8, 'detector_type': DetectorType.black_box.name},
            {'snr': 10, 'detector_type': DetectorType.black_box.name},
            {'snr': 12, 'detector_type': DetectorType.black_box.name},
            {'snr': 4, 'detector_type': DetectorType.bayesian.name},
            {'snr': 6, 'detector_type': DetectorType.bayesian.name},
            {'snr': 8, 'detector_type': DetectorType.bayesian.name},
            {'snr': 10, 'detector_type': DetectorType.bayesian.name},
            {'snr': 12, 'detector_type': DetectorType.bayesian.name},
            {'snr': 4, 'detector_type': DetectorType.deepsic.name},
            {'snr': 6, 'detector_type': DetectorType.deepsic.name},
            {'snr': 8, 'detector_type': DetectorType.deepsic.name},
            {'snr': 10, 'detector_type': DetectorType.deepsic.name},
            {'snr': 12, 'detector_type': DetectorType.deepsic.name},
            {'snr': 4, 'detector_type': DetectorType.model_based_bayesian.name},
            {'snr': 6, 'detector_type': DetectorType.model_based_bayesian.name},
            {'snr': 8, 'detector_type': DetectorType.model_based_bayesian.name},
            {'snr': 10, 'detector_type': DetectorType.model_based_bayesian.name},
            {'snr': 12, 'detector_type': DetectorType.model_based_bayesian.name},
        ]
        values = list(range(8, 17, 2))
        xlabel, ylabel = 'SNR', 'SER'
    elif plot_type == PlotType.MIMO_BY_RELIABILITY_EightPSK:
        params_dicts = [
            {'snr': 10, 'detector_type': DetectorType.bayesian.name},
            {'snr': 10, 'detector_type': DetectorType.model_based_bayesian.name},
            {'snr': 10, 'detector_type': DetectorType.deepsic.name},
        ]
        values = np.linspace(start=0.5, stop=1, num=6)
        xlabel, ylabel = 'Confidence', 'Accuracy/Confidence'
    else:
        raise ValueError('No such plot type!!!')

    return params_dicts, values, xlabel, ylabel
