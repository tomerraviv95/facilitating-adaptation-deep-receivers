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
3. channel_model - chooses the channel taps values, either synthetic or based on COST2100. String in the set ('Cost2100','Synthetic').


4. iterations - number of iterations in the unfolded DeepSIC architecture. Integer.
5. info_size - number of information bits in each training pilot block and test data block. Integer.
6. train_frame_num - number of blocks used for training. Integer.
7. test_frame_num - number of blocks used for test. Integer.
8. test_pilot_size - number of bits in each test pilot block. Integer.
9. fading - whether to use fading. Relevant only to the SED channel. Boolean flag.
10. channel_mode - choose the Spatial Exponential Decay Channel Model, i.e. exp(-|i-j|), or the beamformed COST channel. String value: 'SED' or 'COST'. COST works with 8x8 n_user and n_ant only.
11. lr - learning rate for training. Float.
12. max_epochs - number of offline training epochs. Integer.
13. self_supervised_epochs - number of online training epochs. Integer.
14. use_ecc - whether to use Error Correction Codes (ECC) or not. If not - automatically will run in evaluations the online pilots-blocks scenario (as in pilots efficiency part). Boolean flag.
15. n_ecc_symbols - number of symbols in ecc. Number of additional transmitted bits is 8 times this value, due to the specific Reed-Solomon we employ. [Read more here](https://en.wikiversity.org/wiki/Reed%E2%80%93Solomon_codes_for_coders). Integer.
16. ber_thresh - threshold for self-supervised training, as in the [ViterbiNet](https://arxiv.org/abs/1905.10750) or [DeepSIC](https://arxiv.org/abs/2002.03214) papers.
17. change_user_only - allows change of channel for a single user. Integer value (the index of the desired user: {0,..,n_user}).
18. retrain_user - only in the DeepSIC architecture, allow for training the specific user networks only. Integer value (the index of the desired user: {0,..,n_user}).

## resources

Keeps the COST channel coefficients vectors in 4 test folders. Also holds config runs for the paper.

## dir_definitions 

Definitions of relative directories.
