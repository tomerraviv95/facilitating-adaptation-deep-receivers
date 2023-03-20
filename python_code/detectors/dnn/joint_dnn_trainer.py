from python_code.detectors.dnn.dnn_trainer import DNNTrainer


class JointDNNTrainer(DNNTrainer):
    def __init__(self):
        super().__init__()
        self.is_joint_training = True
