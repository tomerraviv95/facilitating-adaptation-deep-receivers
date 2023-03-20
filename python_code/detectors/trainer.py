import math
import random
from typing import List, Tuple

import numpy as np
import torch
from torch.nn import CrossEntropyLoss, MSELoss
from torch.optim import RMSprop, Adam, SGD

from python_code import DEVICE, conf
from python_code.augmentations.augmenter_wrapper import AugmenterWrapper
from python_code.channel.channel_dataset import ChannelModelDataset
from python_code.channel.modulator import MODULATION_NUM_MAPPING
from python_code.utils.constants import ModulationType
from python_code.utils.metrics import calculate_ber
from python_code.utils.probs_utils import get_bits_from_qpsk_symbols

random.seed(conf.seed)
torch.manual_seed(conf.seed)
torch.cuda.manual_seed(conf.seed)
np.random.seed(conf.seed)

META_BLOCKS_NUM = 5


class Trainer(object):
    """
    Implements the meta-trainer class. Every trainer must inherent from this base class.
    It implements the evaluation method, initializes the dataloader and the detector.
    It also defines some functions that every inherited trainer must implement.
    """

    def __init__(self):
        self.constellation_bits = int(math.log2(MODULATION_NUM_MAPPING[conf.modulation_type]))
        # initialize matrices, datasets and detector
        self.is_online_meta = False
        self.is_online_training = False
        self.is_joint_training = False
        self._initialize_dataloader()
        self._initialize_detector()
        self.softmax = torch.nn.Softmax(dim=1)  # Single symbol probability inference

    def get_name(self):
        return self.__name__()

    def _initialize_detector(self):
        """
        Every trainer must have some base detector deepsic
        """
        self.detector = None

    # calculate train loss
    def calc_loss(self, est: torch.Tensor, tx: torch.Tensor) -> torch.Tensor:
        """
         Every trainer must have some loss calculation
        """
        pass

    # setup the optimization algorithm
    def deep_learning_setup(self, lr: float):
        """
        Sets up the optimizer and loss criterion
        """
        if conf.optimizer_type == 'Adam':
            self.optimizer = Adam(filter(lambda p: p.requires_grad, self.detector.parameters()),
                                  lr=lr)
        elif conf.optimizer_type == 'RMSprop':
            self.optimizer = RMSprop(filter(lambda p: p.requires_grad, self.detector.parameters()),
                                     lr=lr)
        elif conf.optimizer_type == 'SGD':
            self.optimizer = SGD(filter(lambda p: p.requires_grad, self.detector.parameters()),
                                 lr=lr)
        else:
            raise NotImplementedError("No such optimizer implemented!!!")
        if conf.loss_type == 'CrossEntropy':
            self.criterion = CrossEntropyLoss().to(DEVICE)
        elif conf.loss_type == 'MSE':
            self.criterion = MSELoss().to(DEVICE)
        else:
            raise NotImplementedError("No such loss function implemented!!!")

    # setup the optimization algorithm
    def calibration_deep_learning_setup(self):
        """
        Sets up the optimizer and loss criterion
        """
        if conf.optimizer_type == 'Adam':
            self.optimizer = Adam(filter(lambda p: p.requires_grad, self.detector.net.dropout_logit),
                                  lr=self.lr)
        elif conf.optimizer_type == 'RMSprop':
            self.optimizer = RMSprop(filter(lambda p: p.requires_grad, self.detector.net.dropout_logit),
                                     lr=self.lr)
        elif conf.optimizer_type == 'SGD':
            self.optimizer = SGD(filter(lambda p: p.requires_grad, self.detector.net.dropout_logit),
                                 lr=self.lr)
        else:
            raise NotImplementedError("No such optimizer implemented!!!")
        if conf.loss_type == 'CrossEntropy':
            self.criterion = CrossEntropyLoss().to(DEVICE)
        elif conf.loss_type == 'MSE':
            self.criterion = MSELoss().to(DEVICE)
        else:
            raise NotImplementedError("No such loss function implemented!!!")

    def _initialize_dataloader(self):
        """
        Sets up the data loader - a generator from which we draw batches, in iterations
        """
        self.test_channel_dataset = ChannelModelDataset(block_length=conf.block_length,
                                                        pilots_length=conf.pilot_size,
                                                        blocks_num=conf.blocks_num,
                                                        fading_in_channel=conf.fading_in_channel)
        self.train_channel_dataset = ChannelModelDataset(block_length=conf.joint_block_length,
                                                         pilots_length=conf.joint_pilot_size,
                                                         blocks_num=conf.joint_blocks_num,
                                                         fading_in_channel=False)

    def _online_training(self, tx: torch.Tensor, rx: torch.Tensor):
        """
        Every detector trainer must have some function to adapt it online
        """
        pass

    def forward(self, rx: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Every trainer must have some forward pass for its detector
        """
        pass

    def evaluate(self) -> Tuple[List[float], List[float], List[float]]:
        """
        The online evaluation run. Main function for running the experiments of sequential transmission of pilots and
        data blocks for the paper.
        :return: list of ber per timestep
        """
        total_ber = []
        # augmentations
        augmenter_wrapper = AugmenterWrapper(conf.aug_type, conf.fading_in_channel)
        # meta-training's saved detector - saved detector is used to initialize the decoder in meta learning loops
        saved_detector = self.copy_model(self.detector)
        # if in joint training mode, train on the train dataset
        if self.is_joint_training:
            # draw words for a given snr
            joint_transmitted_words, joint_received_words, joint_hs = self.train_channel_dataset.__getitem__(
                snr_list=[conf.joint_snr])
            # get current word and channel
            joint_tx, joint_h, joint_rx = joint_transmitted_words[0], joint_hs[0], joint_received_words[0]
            self._online_training(joint_tx, joint_rx)
        # draw words for a given snr
        transmitted_words, received_words, hs = self.test_channel_dataset.__getitem__(snr_list=[conf.snr])
        # buffer for words and their target
        buffer_tx, buffer_rx = torch.empty([0, transmitted_words.shape[2]]).to(DEVICE), torch.empty(
            [0, received_words.shape[2]]).to(DEVICE)
        # detect sequentially
        for block_ind in range(conf.blocks_num):
            print('*' * 20)
            # get current word and channel
            tx, h, rx = transmitted_words[block_ind], hs[block_ind], received_words[block_ind]
            # split words into data and pilot part
            tx_pilot, tx_data = tx[:conf.pilot_size // self.constellation_bits], \
                                tx[conf.pilot_size // self.constellation_bits:]
            rx_pilot, rx_data = rx[:conf.pilot_size // self.constellation_bits], \
                                rx[conf.pilot_size // self.constellation_bits:]
            # add to buffer
            buffer_tx = torch.cat([buffer_tx, tx_pilot], dim=0)
            buffer_rx = torch.cat([buffer_rx, rx_pilot], dim=0)
            # meta-learning main function
            if self.is_online_meta and conf.fading_in_channel and block_ind > 0 and block_ind % META_BLOCKS_NUM == 0:
                print('Meta')
                self._meta_training(saved_detector, buffer_tx, buffer_rx)

            # online training main function
            if self.is_online_training:
                if self.is_online_meta and block_ind >= META_BLOCKS_NUM:
                    self.detector = self.copy_model(saved_detector)
                # augment received words by the number of desired repeats
                augmenter_wrapper.update_hyperparams(rx_pilot, tx_pilot)
                y_aug, x_aug = augmenter_wrapper.augment_batch(rx_pilot, tx_pilot)
                # re-train the detector
                self._online_training(x_aug, y_aug)
            # detect data part after training on the pilot part
            detected_word = self.forward(rx_data)
            # calculate accuracy
            target = tx_data[:, :rx.shape[1]]
            if conf.modulation_type == ModulationType.QPSK.name:
                target = get_bits_from_qpsk_symbols(target)
            ber = calculate_ber(detected_word, target)
            total_ber.append(ber)
            print(f'current: {block_ind, ber}')
        print(f'Final ser: {sum(total_ber) / len(total_ber)}')
        return total_ber

    def run_train_loop(self, est: torch.Tensor, tx: torch.Tensor) -> float:
        # calculate loss
        loss = self.calc_loss(est=est, tx=tx)
        current_loss = loss.item()
        # back propagation
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return current_loss

    def _meta_training(self, saved_detector, tx_pilot: torch.Tensor, rx_pilot: torch.Tensor):
        pass

    def copy_model(self, detector):
        pass
