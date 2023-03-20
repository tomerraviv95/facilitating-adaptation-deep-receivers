import os

from python_code import conf
from python_code.detectors.deepsic.joint_deepsic_trainer import JointDeepSICTrainer
from python_code.detectors.deepsic.online_deepsic_trainer import OnlineDeepSICTrainer
from python_code.detectors.dnn.joint_dnn_trainer import JointDNNTrainer
from python_code.detectors.dnn.online_dnn_trainer import OnlineDNNTrainer
from python_code.detectors.meta_deepsic.meta_deep_sic_trainer import MetaDeepSICTrainer
from python_code.utils.constants import DetectorType

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
CHANNEL_TYPE_TO_TRAINER_DICT = {
    DetectorType.joint_black_box.name: JointDNNTrainer,
    DetectorType.online_black_box.name: OnlineDNNTrainer,
    DetectorType.joint_deepsic.name: JointDeepSICTrainer,
    DetectorType.online_deepsic.name: OnlineDeepSICTrainer,
    DetectorType.meta_deepsic.name: MetaDeepSICTrainer,
}

if __name__ == '__main__':
    trainer = CHANNEL_TYPE_TO_TRAINER_DICT[conf.detector_type]()
    print(trainer)
    trainer.evaluate()
