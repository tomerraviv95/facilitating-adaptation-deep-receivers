from python_code.detectors.vnet.vnet_trainer import ViterbiNetTrainer


class JointViterbiNetTrainer(ViterbiNetTrainer):
    def __init__(self):
        super().__init__()
        self.is_joint_training = True
