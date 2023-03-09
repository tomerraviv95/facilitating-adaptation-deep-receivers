import os

from python_code import conf
from python_code.detectors.deepsic.deepsic_trainer import DeepSICTrainer
from python_code.detectors.dnn.dnn_trainer import DNNTrainer
from python_code.utils.constants import DetectorType

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

CHANNEL_TYPE_TO_TRAINER_DICT = {DetectorType.deepsic.name: DeepSICTrainer,
                                DetectorType.black_box.name: DNNTrainer}

if __name__ == '__main__':
    trainer = CHANNEL_TYPE_TO_TRAINER_DICT[conf.detector_type]()
    print(trainer)
    trainer.evaluate()
