import datetime
import os
from itertools import chain
from typing import List, Tuple, Dict

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

from dir_definitions import FIGURES_DIR, PLOTS_DIR
from python_code import conf
from python_code.detectors.trainer import Trainer
from python_code.utils.python_utils import load_pkl, save_pkl

mpl.rcParams['xtick.labelsize'] = 24
mpl.rcParams['ytick.labelsize'] = 24
mpl.rcParams['font.size'] = 8
mpl.rcParams['figure.autolayout'] = True
mpl.rcParams['figure.figsize'] = [9.5, 6.45]
mpl.rcParams['axes.titlesize'] = 28
mpl.rcParams['axes.labelsize'] = 28
mpl.rcParams['lines.linewidth'] = 2
mpl.rcParams['lines.markersize'] = 8
mpl.rcParams['legend.fontsize'] = 20
mpl.rcParams['mathtext.fontset'] = 'stix'
mpl.rcParams['font.family'] = 'STIXGeneral'


def get_linestyle(method_name: str) -> str:
    linestyles_dict = {
        'Augmented Online Meta-DeepSIC': 'solid',
        'Augmented Online Meta-ViterbiNet': 'solid',
        'Online Meta-DeepSIC': 'dashdot',
        'Online Meta-ViterbiNet': 'dashdot',
        'Online DeepSIC': 'dashed',
        'Online ViterbiNet': 'dashed',
        'Bayesian Joint DeepSIC': 'dashed',
        'Bayesian Online ViterbiNet': 'dashed',
        'Joint DeepSIC': 'dashed',
        'Joint ViterbiNet': 'dashed',
        'Online RNN Detector': 'dotted',
        'Online DNN Detector': 'dotted',
        'Joint RNN Detector': ':',
        'Joint DNN Detector': ':'
    }
    return linestyles_dict[method_name]


def get_marker(method_name: str) -> str:
    markers_dict = {
        'Augmented Online Meta-DeepSIC': 'o',
        'Augmented Online Meta-ViterbiNet': 'o',
        'Online Meta-DeepSIC': 'X',
        'Online Meta-ViterbiNet': 'X',
        'Online DeepSIC': 's',
        'Online ViterbiNet': 's',
        'Bayesian Joint DeepSIC': 'P',
        'Bayesian Online ViterbiNet': 'P',
        'Joint DeepSIC': 'p',
        'Joint ViterbiNet': 'p',
        'Online DNN Detector': '.',
        'Online RNN Detector': '.',
        'Joint DNN Detector': '8',
        'Joint RNN Detector': '8'
    }
    return markers_dict[method_name]


def get_color(method_name: str) -> str:
    colors_dict = {
        'Augmented Online Meta-DeepSIC': 'black',
        'Augmented Online Meta-ViterbiNet': 'black',
        'Online Meta-DeepSIC': 'blue',
        'Online Meta-ViterbiNet': 'blue',
        'Online DeepSIC': 'red',
        'Online ViterbiNet': 'red',
        'Bayesian Joint DeepSIC': 'pink',
        'Bayesian Online ViterbiNet': 'pink',
        'Joint DeepSIC': 'green',
        'Joint ViterbiNet': 'green',
        'Online DNN Detector': 'cyan',
        'Online RNN Detector': 'cyan',
        'Joint DNN Detector': 'orange',
        'Joint RNN Detector': 'orange'
    }
    return colors_dict[method_name]


def get_all_plots(dec: Trainer, run_over: bool, method_name: str, trial=None):
    print(method_name)
    # set the path to saved plot results for a single method (so we do not need to run anew each time)
    if not os.path.exists(PLOTS_DIR):
        os.makedirs(PLOTS_DIR)
    file_name = '_'.join([method_name, str(conf.channel_type)])
    if trial is not None:
        file_name = file_name + '_' + str(trial)
    plots_path = os.path.join(PLOTS_DIR, file_name)
    print(plots_path)
    # if plot already exists, and the run_over flag is false - load the saved plot
    if os.path.isfile(plots_path + '.pkl') and not run_over:
        print("Loading plots")
        ber_total = load_pkl(plots_path)
    else:
        # otherwise - run again
        print("Calculating fresh")
        ber_total = dec.evaluate()
        save_pkl(plots_path, ber_total)
    return ber_total


def plot_by_values(all_curves: List[Tuple[np.ndarray, np.ndarray, str]], values: List[float], xlabel: str,
                   ylabel: str):
    # path for the saved figure
    current_day_time = datetime.datetime.now()
    folder_name = f'{current_day_time.month}-{current_day_time.day}-{current_day_time.hour}-{current_day_time.minute}'
    if not os.path.isdir(os.path.join(FIGURES_DIR, folder_name)):
        os.makedirs(os.path.join(FIGURES_DIR, folder_name))

    # extract names from simulated plots - backup
    plt.figure()
    names = []
    for i in range(len(all_curves)):
        if all_curves[i][0] not in names:
            names.append(all_curves[i][0])

    cur_name, sers_dict = get_to_plot_values_dict(all_curves, names)
    MARKER_EVERY = 1
    x_ticks = values
    x_labels = values

    # plots - backup all methods
    for method_name in names:
        print(method_name)
        plt.plot(values, sers_dict[method_name], label=method_name,
                 color=get_color(method_name),
                 marker=get_marker(method_name), markersize=11,
                 linestyle=get_linestyle(method_name), linewidth=2.2,
                 markevery=MARKER_EVERY)

    plt.xticks(ticks=x_ticks, labels=x_labels)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.grid(which='both', ls='--')
    plt.legend(loc='lower left', prop={'size': 15})
    plt.yscale('log')
    trainer_name = cur_name.split(' ')[0]
    plt.savefig(os.path.join(FIGURES_DIR, folder_name, f'ser_versus_snrs_{trainer_name}.png'),
                bbox_inches='tight')


def get_to_plot_values_dict(all_curves: List[Tuple[float, str]], names: List[str]) -> Tuple[
    str, Dict[str, List[np.ndarray]]]:
    values_to_plot_dict = {}
    for method_name in names:
        values_to_plot = []
        for cur_name, ser in all_curves:
            if cur_name != method_name:
                continue
            mean_ser = np.mean(np.array(list(chain.from_iterable(ser))))
            values_to_plot.append(mean_ser)
        values_to_plot_dict[method_name] = values_to_plot
    return cur_name, values_to_plot_dict
