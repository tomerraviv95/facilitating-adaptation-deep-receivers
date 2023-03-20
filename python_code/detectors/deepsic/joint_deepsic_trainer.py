from python_code.detectors.deepsic.deepsic_trainer import DeepSICTrainer


class JointDeepSICTrainer(DeepSICTrainer):
    def __init__(self):
        super().__init__()
        self.is_joint_training = True