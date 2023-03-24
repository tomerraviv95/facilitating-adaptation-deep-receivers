from random import randint

import torch

from python_code.detectors.rnn.rnn_detector import RNNDetector
from python_code.detectors.trainer import Trainer
from python_code.utils.config_singleton import Config
from python_code.utils.probs_utils import calculate_siso_states

from python_code.utils.constants import ModulationType
from python_code.utils.probs_utils import calculate_symbols_from_states, get_bits_from_qpsk_symbols

conf = Config()
EPOCHS = 400
BATCH_SIZE = 32


class RNNTrainer(Trainer):
    """
    Trainer for the RNNTrainer model.
    """

    def __init__(self):
        self.memory_length = conf.memory_length
        self.n_states = 2 ** self.memory_length
        self.n_user = 1
        self.n_ant = 1
        self.lr = 5e-3
        super().__init__()

    def __str__(self):
        name = 'RNN Detector'
        if self.is_joint_training:
            name = 'Joint ' + name
        if self.is_online_training:
            name = 'Online ' + name
        return name

    def _initialize_detector(self):
        """
        Loads the RNN detector
        """
        self.detector = RNNDetector(self.memory_length)

    def calc_loss(self, est: torch.Tensor, tx: torch.IntTensor) -> torch.Tensor:
        """
        Cross Entropy loss - distribution over states versus the gt state label
        :param est: [1, transmission_length,n_states], each element is a probability
        :param tx: [1, transmission_length]
        :return: loss value
        """
        gt_states = calculate_siso_states(self.memory_length, tx)
        loss = self.criterion(input=est, target=gt_states)
        return loss

    def forward(self, rx: torch.Tensor, probs_vec: torch.Tensor = None) -> torch.Tensor:
        if conf.modulation_type == ModulationType.BPSK.name:
            rx = rx.float()
        elif conf.modulation_type in [ModulationType.QPSK.name]:
            rx = torch.view_as_real(rx).float().reshape(rx.shape[0], -1)

        soft_estimation = self.detector(rx)
        estimated_states = torch.argmax(soft_estimation, dim=1)
        estimated_words = calculate_symbols_from_states(2 ** self.memory_length, estimated_states)
        detected_word= estimated_words[:, 0].reshape(-1, 1).long()

        if conf.modulation_type == ModulationType.QPSK.name:
            detected_word = get_bits_from_qpsk_symbols(detected_word)
        return detected_word

    def _online_training(self, tx: torch.Tensor, rx: torch.Tensor):
        """
        Online training module - trains on the detected word.
        Start from the previous weights, or from scratch.
        :param tx: transmitted word
        :param rx: received word
        """
        if not conf.fading_in_channel:
            self._initialize_detector()
        self.deep_learning_setup(self.lr)

        if conf.modulation_type in [ModulationType.QPSK.name]:
            rx = torch.view_as_real(rx).float().reshape(rx.shape[0], -1)

        # run training loops
        loss = 0
        for i in range(EPOCHS):
            word_ind = randint(a=0, b=conf.online_repeats_n)
            subword_ind = randint(a=0, b=conf.pilot_size - BATCH_SIZE)
            ind = word_ind * conf.pilot_size + subword_ind
            # pass through detector
            soft_estimation = self.detector(rx[ind: ind + BATCH_SIZE].float())
            current_loss = self.run_train_loop(est=soft_estimation,tx=tx[ind:ind + BATCH_SIZE])
            loss += current_loss
