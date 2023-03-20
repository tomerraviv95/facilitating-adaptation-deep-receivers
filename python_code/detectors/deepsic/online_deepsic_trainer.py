from python_code.detectors.deepsic.deepsic_trainer import DeepSICTrainer


class OnlineDeepSICTrainer(DeepSICTrainer):
    def __init__(self):
        super().__init__()
        self.is_online_training = True
