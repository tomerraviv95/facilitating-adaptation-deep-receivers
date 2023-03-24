from python_code.detectors.rnn.rnn_trainer import RNNTrainer


class JointRNNTrainer(RNNTrainer):
    def __init__(self):
        super().__init__()
        self.is_joint_training = True
