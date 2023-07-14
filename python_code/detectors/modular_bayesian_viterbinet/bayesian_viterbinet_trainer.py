import torch

from python_code.detectors.modular_bayesian_viterbinet.bayesian_viterbinet_detector import BayesianVNETDetector, \
    LossVariable
from python_code.detectors.viterbinet.viterbinet_trainer import ViterbiNetTrainer
from python_code.utils.config_singleton import Config
from python_code.utils.constants import Phase, HALF
from python_code.utils.probs_utils import calculate_siso_states

conf = Config()
EPOCHS = 500


class BayesianVNETTrainer(ViterbiNetTrainer):
    """
    Trainer for the ViterbiNet model.
    """

    def __init__(self):
        self.probs_vec = None
        self.ensemble_num = 5
        self.kl_scale = 5
        self.kl_beta = 1e-3
        self.arm_beta = 2
        super().__init__()
        self.is_bayesian = True
        self.is_joint_training = True

    def __str__(self):
        return 'Bayesian ViterbiNet'

    def _initialize_detector(self):
        """
        Loads the Bayesian ViterbiNet detector
        """
        self.detector = BayesianVNETDetector(n_states=self.n_states,
                                             kl_scale=self.kl_scale,
                                             ensemble_num=self.ensemble_num)

    def calc_loss(self, est: LossVariable, tx: torch.IntTensor) -> torch.Tensor:
        """
        Cross Entropy loss - distribution over states versus the gt state label
        :param est[0]: [1,transmission_length,n_states], each element is a probability
        :est[1]: log_prob_ARM_ori
        :est[2]: log_prob_ARM_tilde
        :est[3]: kl_term
        :est[4]: (u1, u2)
        :param tx: [1, transmission_length]
        :return: loss value
        """
        gt_states = calculate_siso_states(self.memory_length, tx)
        data_fitting_loss_term = self.criterion(input=est.priors, target=gt_states)
        loss = data_fitting_loss_term
        # ARM Loss
        arm_loss = 0
        for i in range(self.ensemble_num):
            loss_term_arm_original = self.criterion(input=est.arm_original[i], target=gt_states)
            loss_term_arm_tilde = self.criterion(input=est.arm_tilde[i], target=gt_states)
            arm_delta = (loss_term_arm_tilde - loss_term_arm_original)
            grad_logit = arm_delta * (est.u_list[i] - HALF)
            arm_loss += torch.matmul(grad_logit, self.detector.net.dropout_logit.T)
        arm_loss = torch.mean(arm_loss)
        # KL Loss
        kl_term = self.kl_beta * est.kl_term
        loss += self.arm_beta * arm_loss + kl_term
        return loss

    def forward(self, rx: torch.Tensor, probs_vec: torch.Tensor = None, h: torch.Tensor = None) -> torch.Tensor:
        # detect and decode
        detected_word = self.detector(rx.float(), phase=Phase.TEST)
        return detected_word

    def _online_training(self, tx: torch.Tensor, rx: torch.Tensor):
        """
        Online training module - trains on the detected word.
        Start from the saved meta-trained weights.
        :param tx: transmitted word
        :param rx: received word
        :param h: channel coefficients
        """
        self._initialize_detector()
        self.deep_learning_setup(self.lr)

        # run training loops
        loss = 0
        for i in range(EPOCHS):
            # pass through detector
            soft_estimation = self.detector(rx.float(), phase=Phase.TRAIN)
            current_loss = self.run_train_loop(est=soft_estimation, tx=tx)
            loss += current_loss
