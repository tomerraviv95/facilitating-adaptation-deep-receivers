*"A man, remember, whether rich or poor, should do something in this world. No one can find happiness without work."* 

--The Adventures of Pinocchio (by Carlo Collodi).

# Adaptive and Flexible Model-Based AI for Deep Receivers in Dynamic Channels

Python repository for the magazine paper "Adaptive and Flexible Model-Based AI for Deep Receivers in Dynamic Channels".

Please cite our [paper](https://arxiv.org/abs/2203.14359), if the code is used for publishing research.

# Table of Contents

- [Introduction](#introduction)
- [Folders Structure](#folders-structure)
  * [python_code](#python_code)
    + [augmentations](#augmentations)
    + [channel](#channel)
    + [detectors](#detectors)
    + [plotters](#plotters)
    + [trainers](#trainers)
    + [utils](#utils)
  * [resources](#resources)
  * [dir_definitions](#dir_definitions)
- [Execution](#execution)
  * [Environment Installation](#environment-installation)

<small><i><a href='http://ecotrust-canada.github.io/markdown-toc/'>Table of contents generated with markdown-toc</a></i></small>

# Introduction

This repository implements deep receivers, with a bunch of different improvements, for SISO and MIMO channels. 
The deep receviers include the ViterbiNet equalizer and an RNN detector for the SISO case, and the DeepSIC receiver 
as well as a fully connected black-box network for the MIMO case. 
It includes the possibility of adding the communication drive augmentations of [my previous augmentations paper](https://arxiv.org/pdf/2209.01362.pdf),
see [this github repo](https://github.com/tomerraviv95/data-augmentations-for-receivers) for the code implementation.
From the training prespective, you can choose between joint training (the receiver is trained offline in a pre-test phase, using data simulated from multitude of channel realizations. 
No additional training is done online in the test phase), online training (the receiver is trained online using the pilots batch at each time step) and meta-learning training taken
from [my previous meta-learning paper](https://arxiv.org/pdf/2203.14359.pdf), code implementation for [siso](https://github.com/tomerraviv95/meta-viterbinet) and [mimo](https://github.com/tomerraviv95/meta-deepsic). In the paper we show that the gains from the different approaches add up individually over the two scenarios and a range of SNRs.
  

# Folders Structure

## python_code 

The python simulations of the simplified communication chain: symbols generation, channel transmission and detection.

### augmentations

The proposed augmentations scheme suggested in [my previous augmentations paper](https://arxiv.org/pdf/2209.01362.pdf),
see [this github repo](https://github.com/tomerraviv95/data-augmentations-for-receivers) for the code implementation.

### channel 

Includes all relevant channel functions and classes. The class in "channel_dataset.py" implements the main class for aggregating pairs of (transmitted,received) samples. 
In "channel.py", the SISO and MIMO channels are implemented. "channel_estimation.py" is for the calculation of the h values. Lastly, the channel BPSK/QPSK modulators lies in "channel_modulator.py".

### detectors

The backbone detectors and their respective training: ViterbiNet, DeepSIC, Meta-ViterbiNet, Meta-DeepSIC, RNN BlackBox and FC BlackBox. The meta and non-meta detectors have slightly different API so they are seperated in the trainer class below. The meta variation has to receive the parameters of the original architecture for the training. The trainers are wrappers for the training and evaluation of the detectors. Trainer holds the training, sequential evaluation of pilot + info blocks. It also holds the main function 'eval' that trains the detector and evaluates it, returning a list of coded ber/ser per block. The train and test dataloaders are also initialized by the trainer. Each trainer is executable by running the 'evaluate.py' script after adjusting the config.yaml hyperparameters and choosing the desired method.

### plotters

Features main plotting tools for the paper:

* plotter_main - main plotting script used to get the figures in the paper. Based on the chosen PlotType enum loads the relevant config and runs the experiment.
* plotter_config - holds a mapping from the PlotType enum to the experiment's hyperparameters and setup.
* plotter_utils - colors, markers and linestyles for all evaluated methods, and the main plotting functions.
* plotter_methods - additional methods used for plotting.

### utils

Extra utils for pickle manipulations and tensor reshaping; calculating the accuracy over FER and BER; several constants; and the config singleton class.
The config works by the [singleton design pattern](https://en.wikipedia.org/wiki/Singleton_pattern). Check the link if unfamiliar. 

The config is accessible from every module in the package, featuring the next parameters:
1. seed - random number generator seed. Integer.
2. channel_type - run either siso or mimo setup. Values in the set of ['SISO','MIMO']. String.
3. channel_model - chooses the channel taps values, either synthetic or based on COST2100. String in the set ['Cost2100','Synthetic'].
4. detector_type - selects the training + architecture to run. Short description of each option: 
* 'joint_black_box - Joint training of the black-box fully connected in the MIMO case.
* 'online_black_box' - Joint training of the black-box fully connected in the MIMO case.
* 'joint_deepsic' - Joint training of the black-box fully connected in the MIMO case.
* 'online_deepsic' - Joint training of the black-box fully connected in the MIMO case.
* 'meta_deepsic' - Joint training of the black-box fully connected in the MIMO case.
* 'joint_rnn' - Joint training of the black-box fully connected in the MIMO case.
* 'online_rnn' - Joint training of the black-box fully connected in the MIMO case.
* 'joint_viterbinet' - Joint training of the black-box fully connected in the MIMO case.
* online_viterbinet' - Joint training of the black-box fully connected in the MIMO case.
* 'meta_viterbinet' - Joint training of the black-box fully connected in the MIMO case.
5. linear - whether to apply non-linear tanh at the channel output, not used in the paper but still may be applied. Bool.
6.fading_in_channel - whether to use fading. Relevant only to the synthetic channel. Boolean flag.
7. snr - signal-to-noise ratio, determines the variance properties of the noise, in dB. Float.
8. modulation_type - either 'BPSK' or 'QPSK', string.
9. memory_length - siso channel hyperparameter, integer.
10. n_user - mimo channel hyperparameter, number of transmitting devices. Integer.
11. n_ant - mimo channel hyperparameter, number of receiving devices. Integer.
12. block_length - number of coherence block bits, total size of pilot + data. Integer.
13. pilot_size - number of pilot bits. Integer.
14. blocks_num - number of blocks in the tranmission. Integer.
15. loss_type - 'CrossEntropy', could be altered to other types 'BCE' or 'MSE'.
16. optimizer_type - 'Adam', could be altered to other types 'RMSprop' or 'SGD'.
17. joint_block_length - joint training hyperparameter. Offline training block length. Integer.
18. joint_pilot_size - joint training hyperparameter. Offline training pilots block length. Integer.
19. joint_blocks_num - joint training hyperparameter. Number of blocks to train on offline. Integer.
20. joint_snrs - joint training hyperparameter. Number of SNRs to traing from offline. List of float values.
21. aug_type - what augmentations to use. leave empty list for no augmentations, or add whichever of the following you like: ['geometric_augmenter','translation_augmenter','rotation_augmenter']
22. online_repeats_n - if using augmentations, adds this factor times the number of pilots to the training batch. Leave at 0 if not using augmentations, if using augmentations try integer values in 2-5.

## resources

Keeps the COST channel coefficients vectors. Also holds config runs for the paper's numerical comparisons figures.

## dir_definitions 

Definitions of relative directories.

# Execution

To execute the code, first download and install Git, Anaconda and PyCharm.

Then install the environment, follow the installation setup below. 

At last, open PyCharm in the root directory. You may run either the trainers or one of the plotters.

This code was simulated with GeForce RTX 3060 with CUDA 12. 

## Environment Installation

1. Open git bash and cd to a working directory of you choice.

2. Clone this repository to your local machine.

3. Open Anaconda prompt and navigate to the cloned repository.

4. Run the command "conda env create -f environment.yml". This should install the required python environment.

5. Open the cloned directory using PyCharm.

6. After the project has been opened in PyCharm, go to settings, File -> Settings... (or CTRL ALT S)

7. In the opened window open the tab Project -> Project Interpreter

8. In the new window, click on the cog icon and then on Add...

9. In the add python interpreter window, click on the Conda Environment tab

10. Select Existing environment and navigate to where the python.exe executable of the deep_ensemble environment is installed under the interpreter setting

  - For windows its usually found at C:\users\<username>\anaconda3\envs\environment\python.exe)

  - For linux its usually found at /home/<username>/anaconda3
  
11. Click OK

12. Done!
