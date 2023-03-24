from python_code.detectors.viterbinet.viterbinet_trainer import ViterbiNetTrainer


class JointViterbiNetTrainer(ViterbiNetTrainer):
    def __init__(self):
        super().__init__()
        self.is_joint_training = True
