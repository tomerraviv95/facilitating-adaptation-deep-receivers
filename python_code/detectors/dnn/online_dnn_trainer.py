from python_code.detectors.dnn.dnn_trainer import DNNTrainer


class OnlineDNNTrainer(DNNTrainer):
    def __init__(self):
        super().__init__()
        self.is_online_training = True
