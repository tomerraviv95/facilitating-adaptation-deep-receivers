import os

from python_code import conf
from python_code.detectors.deepsic.joint_deepsic_trainer import JointDeepSICTrainer
from python_code.detectors.deepsic.online_deepsic_trainer import OnlineDeepSICTrainer
from python_code.detectors.dnn.joint_dnn_trainer import JointDNNTrainer
from python_code.detectors.dnn.online_dnn_trainer import OnlineDNNTrainer
from python_code.detectors.meta_deepsic.meta_deep_sic_trainer import MetaDeepSICTrainer
from python_code.detectors.meta_viterbinet.meta_viterbinet_trainer import MetaViterbiNetTrainer
from python_code.detectors.rnn.joint_rnn_trainer import JointRNNTrainer
from python_code.detectors.rnn.online_rnn_trainer import OnlineRNNTrainer
from python_code.detectors.viterbinet.joint_viterbinet_trainer import JointViterbiNetTrainer
from python_code.detectors.viterbinet.online_viterbinet_trainer import OnlineViterbiNetTrainer
from python_code.utils.constants import DetectorType

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
CHANNEL_TYPE_TO_TRAINER_DICT = {
    DetectorType.joint_black_box.name: JointDNNTrainer,
    DetectorType.online_black_box.name: OnlineDNNTrainer,
    DetectorType.joint_deepsic.name: JointDeepSICTrainer,
    DetectorType.online_deepsic.name: OnlineDeepSICTrainer,
    DetectorType.meta_deepsic.name: MetaDeepSICTrainer,
    DetectorType.joint_rnn.name: JointRNNTrainer,
    DetectorType.online_rnn.name: OnlineRNNTrainer,
    DetectorType.joint_viterbinet.name: JointViterbiNetTrainer,
    DetectorType.online_viterbinet.name: OnlineViterbiNetTrainer,
    DetectorType.meta_viterbinet.name: MetaViterbiNetTrainer,
}

if __name__ == '__main__':
    trainer = CHANNEL_TYPE_TO_TRAINER_DICT[conf.detector_type]()
    print(trainer)
    trainer.evaluate()
