*"A man, remember, whether rich or poor, should do something in this world. No one can find happiness without work."* 

--The Adventures of Pinocchio (by Carlo Collodi).

# Adaptive and Flexible Model-Based AI for Deep Receivers in Dynamic Channels

Python repository for the magazine paper "Adaptive and Flexible Model-Based AI for Deep Receivers in Dynamic Channels".

Please cite our [paper](https://arxiv.org/abs/2203.14359), if the code is used for publishing research.

# Table of Contents

- [Introduction](#introduction)
- [Folders Structure](#folders-structure)
  * [python_code](#python_code)
    + [data](#data)
    + [detectors](#detectors)
    + [ecc](#ecc)
    + [plotting](#plotting)
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
see [this github repo](https://github.com/tomerraviv95/data-augmentations-for-receivers) for the original implementation.
From the training prespective, you can choose between joint training (the receiver is trained offline in a pre-test phase, using data simulated from multitude of channel realizations. 
No additional training is done online in the test phase), online training (the receiver is trained online using the pilots batch at each time step) and meta-learning training taken
from [my previous meta-learning paper](https://arxiv.org/pdf/2203.14359.pdf). In the paper we show that the gains from the different approaches add up individually over the two scenarios and a range of SNRs.
  
