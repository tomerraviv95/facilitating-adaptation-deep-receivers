from python_code.detectors.viterbinet.viterbinet_trainer import ViterbiNetTrainer


class OnlineViterbiNetTrainer(ViterbiNetTrainer):
    def __init__(self):
        super().__init__()
        self.is_online_training = True
