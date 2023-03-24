from python_code.detectors.rnn.rnn_trainer import RNNTrainer


class OnlineRNNTrainer(RNNTrainer):
    def __init__(self):
        super().__init__()
        self.is_online_training = True
