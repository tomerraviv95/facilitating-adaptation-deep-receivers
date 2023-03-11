from python_code.plotters.plotter_config import get_config, PlotType
from python_code.plotters.plotter_methods import compute_for_method, RunParams
from python_code.plotters.plotter_utils import plot_by_values

## Plotter for the Paper's Figures
if __name__ == '__main__':
    run_over = False  # whether to run over previous results
    trial_num = 1  # number of trials per point estimate, used to reduce noise by averaging results of multiple runs
    plot_type = PlotType.LINEAR_COST_QPSK  # Choose the plot among the three Figures
    print(plot_type.name)
    run_params_obj = RunParams(run_over=run_over, trial_num=trial_num)
    params_dicts, values, xlabel, ylabel = get_config(plot_type)
    all_curves = []

    for params_dict in params_dicts:
        print(params_dict)
        compute_for_method(all_curves, params_dict, run_params_obj, plot_type.name)

    plot_by_values(all_curves, values, xlabel, ylabel)
